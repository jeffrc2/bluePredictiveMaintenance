import FIFO::*;
import BRAM::*;
import BRAMFIFO::*;
import FIFOF::*;
import Dense::*;

import WeightMem4::*;

import LSTM::*;
//import Dense::*;


//typedef enum {INIT, LOAD,INPUT} State deriving(Bits,Eq);
typedef enum {INIT, LSTM_1, LSTM_2, DENSE} Layer deriving(Bits,Eq);
typedef enum {INIT, KERNEL, RECURRENT, BIAS} Weight deriving(Bits,Eq);
typedef enum {INIT, INPUT, FORGET, CANDIDATE, OUTPUT, DENSE} Subweight deriving(Bits,Eq);
typedef enum {UPPER, LOWER} Mask deriving(Bits,Eq);

interface PredictiveMaintenanceIfc;
        method Action transmitWeight(Bit#(8) data); //send weight to LSTM1/LSTM2/Dense
		method Action transmitInput(Bit#(8) data); //send input to LSTM1
		method Bool uartStatus; //check availability of uart
        method Bit#(3) rgbOut; //Output of RGB
		method Bool loadStatus; //Load status of weights into SPRAM
		
		method Bool uartOutReady; //Output status of PredictiveMaintenance
		
		method ActionValue#(Bit#(8)) uartOut; //Retrieve output of PredictiveMaintenance
endinterface

module mkPredictiveMaintenance(PredictiveMaintenanceIfc);
		
		WeightMem4Ifc weightMem4 <- mkWeightMem4; //Consolidated weight manager
		
		Reg#(Bit#(14)) loadCounter <- mkReg(0); //load counter for loading weights 
		Reg#(Bit#(14)) weight_addr <- mkReg(0); //temporary address manager for storing weights
		
		Reg#(Layer) layerState <- mkReg(LSTM_1); //load layer state
		Reg#(Weight) weightState <- mkReg(KERNEL); //load layer-weight state
		Reg#(Subweight) subweightState <- mkReg(INPUT); //load layer-subweight state
		Reg#(Mask) maskState <- mkReg(UPPER); //load subweight memory mask state
		
		Reg#(Bool) loadComplete <- mkReg(False); //load completion status indicator
		//Reg#(Bool) loadComplete <- mkReg(True); //load completion status indicator
		
		
		FIFOF#(Bit#(8)) uartOutQ <- mkSizedBRAMFIFOF(2); //
		
		LSTMIfc#(50, 25, 100, 100, 0, 1) lstm1 <- mkLSTM; //LSTM1; memory access offset of 0 as starting point.
		
		LSTMIfc#(50, 100, 50, 50, 6300, 0) lstm2<- mkLSTM; //LSTM2; memory access offset of 6300 = 1250 + 5000 + 50 = 10000/8 + 40000/8 + 400/8 based on kernel, recurrent, and bias weights of LSTM1

		DenseIfc#(50, 1, 10075) dense <- mkDense; //Dense; memory access offset of offset 10075 = 6300 + 2500 + 1250 + 25 = offset1 + 20000/8 + 10000/8 + 200/8 based on kernel, recurrent, and bias weights of LSTM1 and LSTM2
		
		Reg#(Bool) reqProcessing <- mkReg(False); //Request status - indicates ongoing memory request
		
		//Rule managing SPRAM requests - Priority order is from last to first.
		rule lstmRequest(loadComplete == True && reqProcessing == False && (lstm1.requestQueued || lstm2.requestQueued || dense.requestQueued));
			if (dense.requestQueued) begin
				//`ifdef BSIM
				//$display("denseRequest");
				//`endif
				Bit#(14) reqAddr <- lstm2.getRequest;
				weightMem4.reqWeight(reqAddr, 2);
				reqProcessing <= True;
			end
			else if (lstm2.requestQueued) begin
				//`ifdef BSIM
				//$display("lstm2Request");
				//`endif
				Bit#(14) reqAddr <- lstm2.getRequest;
				weightMem4.reqWeight(reqAddr, 1);
				reqProcessing <= True;
			end
			else if (lstm1.requestQueued) begin
				//`ifdef BSIM
				//$display("lstm1Request");
				//`endif
				Bit#(14) reqAddr <- lstm1.getRequest;
				weightMem4.reqWeight(reqAddr, 0);
				reqProcessing <= True;
			end
			
		endrule
		
		//Rule retrieving requested memory - Retrieves weights from the SPRAM Manager and sends it to the corresponding module
		rule memRead(loadComplete == True && reqProcessing == True);
			//`ifdef BSIM
			//$display("memRead");
			//`endif
			
			Tuple2#(Bit#(64), Bit#(2)) weights <- weightMem4.recvWeight;
			Bit#(64) wt = tpl_1(weights);
			case (tpl_2(weights)) matches 
				0: begin
					lstm1.processWeight(wt);
					//`ifdef BSIM
					//$display("lstm1");
					//`endif
				end
				1: begin
					lstm2.processWeight(wt);
					`ifdef BSIM
					$display("lstm2");
					`endif
				end
				2: begin
					dense.processWeight(wt);
					`ifdef BSIM
					$display("dense");
					`endif
				end
			endcase
			
			reqProcessing <= False;
		endrule
		
		//Rule transfers output from lstm1 to lstm2 input queue
		rule lstm12Relay(lstm1.outputQueued);
			lstm2.start;
			Int#(8) out <- lstm1.getOutput;
			lstm2.processInput(out);
			`ifdef BSIM
			$display("relay12");
			`endif
		endrule
		
		//Rule transfers output from lstm2 to dense input queue
		rule lstm2DenseRelay(lstm2.outputQueued);
			dense.start;
			Int#(8) out <- lstm2.getOutput;
			dense.processInput(out);	
			`ifdef BSIM
			$display("relay2Dense");
			`endif
		endrule
		
		//Rule transfers output from dense to uart queue.
		rule denseUartRelay(dense.outputQueued);
			Int#(8) out <- dense.getOutput;
			uartOutQ.enq(pack(out));
			`ifdef BSIM
			$display("relayDenseUart");
			`endif
		endrule
		
		//Method sends data to LSTM1 input queue
		method Action transmitInput(Bit#(8) data);
			`ifdef BSIM
			$display("transmitInput %u", data);
			`endif
			lstm1.processInput(unpack(data));
		endmethod
		
		//Method sends weight data to SPRAM
        method Action transmitWeight(Bit#(8) data);
			`ifdef BSIM
			$display("transmitWeight %u", data);
			`endif
			Bit#(4) bytemask = 4'b1111;
			Bit#(16) maskeddata = 0;
			//Bit#(8) filler = 0;
			
			case (maskState) matches //Sets mask for upper or lower byte of the memory
				UPPER: begin
					bytemask = 4'b1100;
					maskeddata[15:8] = data;
				end
				LOWER: begin
					bytemask = 4'b0011;
					maskeddata[7:0] = data;
				end
			endcase
			case (subweightState) matches //Sends the weight write for the specific weightset to the SPRAM manager
				INPUT: begin
					weightMem4.writeWeight(pack(weight_addr), maskeddata, 2'b00, bytemask);
				end
				FORGET: begin
					weightMem4.writeWeight(pack(weight_addr), maskeddata, 2'b01, bytemask);
				end
				CANDIDATE: begin
					weightMem4.writeWeight(pack(weight_addr), maskeddata, 2'b10, bytemask);
				end
				OUTPUT: begin
					weightMem4.writeWeight(pack(weight_addr), maskeddata, 2'b11, bytemask);
				end
			endcase
			
			Bit#(16) sectionLen = 0; //Variable tracks the start and end of the current weight section
			
			//length constants for managing processing
			//let lstm1KernelLenDiv4 = 2500;
			//let lstm1KernelLenDiv8 = 1250;
			//let lstm1RecurrentLenDiv4 = 10000;
			//let lstm1RecurrentLenDiv8 = 5000;
			//let lstm1BiasLenDiv4 = 100;
			//let lstm1BiasLenDiv8 = 25;
			//let lstm2KernelLenDiv4 = 5000;
			//let lstm2KernelLenDiv8 = 2500;
			//let lstm2RecurrentLenDiv4 = 2500;
			//let lstm2RecurrentLenDiv8 = 1250;
			//let lstm2BiasLenDiv4 = 50;
			//let lstm2BiasLenDiv8 = 25;
			//let denseKernelLen = 50;
			//let denseBiasLen = 1;
			
			Bit#(16) lstm1KernelLen = 10000; //Total LSTM1 kernel weight count
			Bit#(16) lstm1RecurrentLen = 40000; //Total LSTM1 recurrent weight count
			Bit#(16) lstm1BiasLen = 400; //Total LSTM1 bias weight count
			Bit#(16) lstm2KernelLen = 20000; //Total LSTM2 kernel weight count
			Bit#(16) lstm2RecurrentLen = 10000; //Total LSTM2 recurrent weight count
			Bit#(16) lstm2BiasLen = 200; //Total LSTM2 bias weight count
			Bit#(16) denseKernelLen = 50; //Total Dense kernel weight count
			Bit#(16) denseBiasLen = 1; //Total Dense bias weight count
			
			Bit#(14) new_addr = weight_addr+1; //increment weight_addr for contiguous weight. (default value only used when processing O weight)
			Bit#(14) sectionStart = 0;
			
			case (layerState) matches //Sets the corresponding case according to the network; sectionLen is for counting the streamed weights, sectionStart is for setting the address of the beginning of the section.
			//Sections: Divide by 4 for the 4 sections, I,F,C,O
			//Addresses: Divide by 8 because of the additional masked address sharing.
				LSTM_1: begin
					case (weightState) matches //Sets sectionLen for the corresponding LSTM1 weight space in the SPRAM
						KERNEL: begin
							//sectionLen = 2500;
							sectionLen = lstm1KernelLen/4;
							sectionStart = 0;
						end
						RECURRENT: begin
							//sectionLen = lstm1RecurrentLen
							sectionLen = lstm1RecurrentLen/4;
							sectionStart = truncate(lstm1KernelLen/8);
						end
						BIAS: begin
							sectionLen = lstm1BiasLen/4;
							sectionStart = truncate(lstm1KernelLen/8 + lstm1RecurrentLen/8);
						end
					endcase
					
				end
				LSTM_2: begin
				
					case (weightState) matches //Sets sectionLen for the corresponding LSTM2 weight space in the SPRAM
						KERNEL: begin
							sectionLen = lstm2KernelLen/4;
							sectionStart = truncate(lstm1KernelLen/8 + lstm1RecurrentLen/8 + lstm1BiasLen/8);
						end
						RECURRENT: begin
							sectionLen = lstm2RecurrentLen/4;
							sectionStart = truncate(lstm1KernelLen/8 + lstm1RecurrentLen/8 + lstm1BiasLen/8 + lstm2KernelLen/8);
						end
						BIAS: begin
							sectionLen = lstm2BiasLen/4;
							sectionStart = truncate(lstm1KernelLen/8 + lstm1RecurrentLen/8 + lstm1BiasLen/8 + lstm2KernelLen/8 + lstm2RecurrentLen/8);
						end
					endcase
				end
				DENSE: begin //Sets sectionLen for the corresponding Dense weight space in the SPRAM
					case (weightState) matches
						KERNEL: sectionLen = denseKernelLen;
						BIAS: sectionLen = denseBiasLen;
					endcase
				end
			endcase
			
			//Load State Machine
			if (loadCounter == truncate(sectionLen) - 1) begin //When the loadCounter reaches current sectionLen it moves to the next weight memory section.
				
				Subweight subUpdate = subweightState;
				Weight wtUpdate = weightState;
				Layer layerUpdate = layerState;
				Bool loadUpdate = loadComplete;
				if (layerState != DENSE) begin
					case (subweightState) matches // For LSTMs, transfers and sets access to the subsequent weight section. I,F,C,O
						INPUT: begin
							subUpdate = FORGET;
							new_addr = sectionStart;
							`ifdef BSIM
								$display("I weights loaded");
							`endif
						end
						FORGET: begin
							subUpdate = CANDIDATE;
							new_addr = sectionStart;
							`ifdef BSIM
								$display("F weights loaded");
							`endif
						end
						CANDIDATE: begin
							subUpdate = OUTPUT;	
							new_addr = sectionStart;
							`ifdef BSIM
								$display("C weights loaded");
							`endif
						end
						OUTPUT: begin
							`ifdef BSIM
								$display("O weights loaded");
							`endif
							
							subUpdate = INPUT;
							case (weightState) matches
								KERNEL: wtUpdate = RECURRENT;
								RECURRENT: wtUpdate = BIAS;
								BIAS: wtUpdate = KERNEL;
							endcase
							case (layerState) matches
								LSTM_1: begin
									layerUpdate = LSTM_2;
									`ifdef BSIM
									$display("LSTM_1 loaded");
									`endif
								end
								LSTM_2: begin
									layerUpdate = DENSE; 
									`ifdef BSIM
									$display("LSTM_2 loaded");
									`endif
								end
							endcase
						end
					endcase
				end
				else begin
					case (weightState) matches // For Dense, transfers and sets access to bias from kernel, or finishes load if bias' end is reached
						KERNEL: wtUpdate = BIAS; 
						BIAS: begin //end of load
							layerUpdate = INIT;
							wtUpdate = INIT;
							subUpdate = INIT;
							loadUpdate = True;
							`ifdef BSIM
							$display("DENSE loaded");
							`endif
							lstm1.start;
						end
					endcase
				end
				loadCounter <= 0; //reset loadCounter 
				
				//update to the corresponding sections
				subweightState <= subUpdate;
				weightState <= wtUpdate;
				layerState <= layerUpdate;
				loadComplete <= loadUpdate;
			end
			else loadCounter <= loadCounter + 1; //increases load counter for contiguous access.
			if (layerState != DENSE) begin //Alternate bytemask for weights sharing same address in LSTM layers, keeps it in upper for Dense layer. Updates weight address for every dense weight, or every other LSTM weight.
				case (maskState) matches
					UPPER: maskState <= LOWER;
					LOWER: begin
						maskState <= UPPER;
						weight_addr <= new_addr;
					end
				endcase
			end else weight_addr <= new_addr;

			//alternate bytemask
			//increase addr every other.
        endmethod
		
		method Bool loadStatus;
			return loadComplete;
		endmethod
		
		method Bool uartOutReady;
			return uartOutQ.notEmpty;
		endmethod
		
		method ActionValue#(Bit#(8)) uartOut;
			uartOutQ.deq;
			Bit#(8) out = uartOutQ.first;
			return out;
		endmethod
		
		method Bool uartStatus;
			if (loadComplete == True) begin
				return lstm1.inputReady;
			end
			else return True;
		endmethod
		
endmodule : mkPredictiveMaintenance
