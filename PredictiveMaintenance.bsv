import FIFO::*;
import BRAM::*;
import BRAMFIFO::*;
import FIFOF::*;
import Dense::*;

import WeightMem4::*;

import LSTM::*;
//import Dense::*;


typedef enum {INIT, LOAD,INPUT} State deriving(Bits,Eq);
typedef enum {INIT, LSTM_1, LSTM_2, DENSE} Layer deriving(Bits,Eq);
typedef enum {INIT, KERNEL, RECURRENT, BIAS} Weight deriving(Bits,Eq);
typedef enum {INIT, INPUT, FORGET, CANDIDATE, OUTPUT, DENSE} Subweight deriving(Bits,Eq);
typedef enum {UPPER, LOWER} Mask deriving(Bits,Eq);

interface PredictiveMaintenanceIfc;
        method Action transmitWeight(Bit#(8) data);
		method Action transmitInput(Bit#(8) data);
		method Bool uartStatus;
        method Bit#(3) rgbOut;
		method Bool loadStatus;
		
		method Bool uartOutReady;
		method ActionValue#(Bit#(8)) uartOut;
endinterface

module mkPredictiveMaintenance(PredictiveMaintenanceIfc);
		
		WeightMem4Ifc weightMem4 <- mkWeightMem4;
		
		Reg#(Bit#(14)) loadCounter <- mkReg(0);
		Reg#(Bit#(14)) weight_addr <- mkReg(0);
		
		Reg#(State) predictMainState <- mkReg(LOAD);
		Reg#(Layer) layerState <- mkReg(LSTM_1);
		Reg#(Weight) weightState <- mkReg(KERNEL);
		Reg#(Subweight) subweightState <- mkReg(INPUT);
		Reg#(Mask) maskState <- mkReg(UPPER);
		Reg#(Bool) loadComplete <- mkReg(False);
		
		Reg#(Bit#(8)) temporaryInputReg <- mkReg(0);

		Bit#(16) lstm1KernelLen = 10000;
		Bit#(16) lstm1RecurrentLen = 40000;
		Bit#(16) lstm1BiasLen = 400;
		Bit#(16) lstm2KernelLen = 20000;
		Bit#(16) lstm2RecurrentLen = 10000;
		Bit#(16) lstm2BiasLen = 200;
		Bit#(16) denseKernelLen = 50;
		Bit#(16) denseBiasLen = 1;
		
		
		
		FIFOF#(Bit#(8)) uartOutQ <- mkSizedFIFOF(2);
		
		LSTMIfc#(50, 25, 100, 100, 0) lstm1 <- mkLSTM;
		LSTMIfc#(50, 100, 50, 50, 6300) lstm2<- mkLSTM; //offset = 10000/8 + 40000/8 + 400/8 = 1250 + 5000 + 50
		DenseIfc#(50, 1, 10075) dense <- mkDense; //offset = offset1 + 20000/8 + 10000/8 + 200/8 
		
		Reg#(Bool) streamComplete <- mkReg(False);
		
		Reg#(Bool) reqProcessing <- mkReg(False);
		
		Reg#(Bit#(14)) testaddr <- mkReg(0);
		
		rule lstmRequest(loadComplete == True && reqProcessing == False && (lstm1.requestQueued || lstm2.requestQueued || dense.requestQueued));
			if (dense.requestQueued) begin
				Bit#(14) reqAddr <- lstm2.getRequest;
				weightMem4.reqWeight(reqAddr, 0);
				reqProcessing <= True;
			end
			else if (lstm2.requestQueued) begin
				Bit#(14) reqAddr <- lstm2.getRequest;
				weightMem4.reqWeight(reqAddr, 0);
				reqProcessing <= True;
			end
			else if (lstm1.requestQueued) begin
				Bit#(14) reqAddr <- lstm1.getRequest;
				weightMem4.reqWeight(reqAddr, 1);
				reqProcessing <= True;
			end
			
		endrule
		
		rule memRead(loadComplete == True && reqProcessing == True);
			Tuple2#(Bit#(64), Bit#(2)) weights <- weightMem4.recvWeight;
			$display("read check");
			case (tpl_2(weights)) matches 
				0: lstm1.processWeight(tpl_1(weights));
				1: lstm2.processWeight(tpl_1(weights));
			endcase
			reqProcessing <= False;
		endrule
		
		rule lstm12Relay(lstm1.outputQueued);
			lstm2.start;
			Int#(8) out <- lstm1.getOutput;
			lstm2.processInput(out);
		endrule
		
		rule lstm2DenseRelay(lstm2.outputQueued);
			dense.start;
			Int#(8) out <- lstm2.getOutput;
			dense.processInput(out);			
		endrule
		
		rule denseUartRelay(dense.outputQueued);
			Int#(8) out <- dense.getOutput;
			uartOutQ.enq(pack(out));
		endrule
		
		
		method Action transmitInput(Bit#(8) data);
			lstm1.processInput(unpack(data));
		endmethod
		
        method Action transmitWeight(Bit#(8) data);
			Bit#(4) bytemask = 4'b1111;
			Bit#(16) maskeddata = ?;
			//Bit#(8) filler = 0;
			case (maskState) matches
				UPPER: begin
					bytemask = 4'b1100;
					maskeddata[15:8] = data;
				end
				LOWER: begin
					bytemask = 4'b0011;
					maskeddata[7:0] = data;
				end
			endcase
			case (subweightState) matches
				INPUT: begin
					weightMem4.writeWeight(pack(weight_addr), maskeddata, 2'b00, bytemask);
				end
				FORGET: begin
					weightMem4.writeWeight(pack(weight_addr), maskeddata, 2'b01, bytemask);
				end
				CANDIDATE: begin
					weightMem4.writeWeight(pack(weight_addr), maskeddata, 2'b01, bytemask);
				end
				OUTPUT: begin
					weightMem4.writeWeight(pack(weight_addr), maskeddata, 2'b11, bytemask);
				end
			endcase
			
			Bit#(16) sectionLen = 0;
			case (layerState) matches
				LSTM_1: begin
					case (weightState) matches
						KERNEL: sectionLen = lstm1KernelLen/4;
						RECURRENT: sectionLen = lstm1RecurrentLen/4;
						BIAS: sectionLen = lstm1BiasLen/4;
						
					endcase
				end
				LSTM_2: begin
					case (weightState) matches
						KERNEL: sectionLen = lstm2KernelLen/4;
						RECURRENT: sectionLen = lstm2RecurrentLen/4;
						BIAS: sectionLen = lstm2BiasLen/4;
					endcase
				end
				DENSE: begin
					case (weightState) matches
						KERNEL: sectionLen = denseKernelLen;
						BIAS: sectionLen = denseBiasLen;
					endcase
				end
			endcase
			
			//Load State Machine
			if (loadCounter == truncate(sectionLen) - 1) begin
				Subweight subUpdate = subweightState;
				Weight wtUpdate = weightState;
				Layer layerUpdate = layerState;
				Bool loadUpdate = loadComplete;
				if (layerState != DENSE) begin
					case (subweightState) matches
						INPUT: subUpdate = FORGET;
						FORGET: subUpdate = CANDIDATE;
						CANDIDATE: subUpdate = OUTPUT;
						OUTPUT: begin
							subUpdate = INPUT;
							case (weightState) matches
								KERNEL: wtUpdate = RECURRENT;
								RECURRENT: wtUpdate = BIAS;
								BIAS: wtUpdate = KERNEL;
							endcase
							case (layerState) matches
								LSTM_1: begin
									layerUpdate = LSTM_2; 
									$display("LSTM_1 loaded");
								end
								LSTM_2: begin
									layerUpdate = DENSE; 
									$display("LSTM_2 loaded");
									
								end
							endcase
						end
					endcase
				end
				else begin
					case (weightState) matches
						KERNEL: wtUpdate = BIAS; 
						BIAS: begin //end of load
							layerUpdate = INIT;
							wtUpdate = INIT;
							subUpdate = INIT;
							loadUpdate = True;
							$display("DENSE loaded");
							lstm1.start;
						end
					endcase
				end
				loadCounter <= 0;
				subweightState <= subUpdate;
				weightState <= wtUpdate;
				layerState <= layerUpdate;
				loadComplete <= loadUpdate;

			end
			else loadCounter <= loadCounter + 1;
			
			//alternate bytemask
			//increase addr every other.
			if (layerState != DENSE) begin
				case (maskState) matches
					UPPER: maskState <= LOWER;
					LOWER: begin
						maskState <= UPPER;
						weight_addr <= weight_addr + 1;
					end
				endcase
			end else weight_addr <= weight_addr + 1;
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
