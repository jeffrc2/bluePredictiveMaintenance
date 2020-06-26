import FIFO::*;
import BRAM::*;
import BRAMFIFO::*;

import WeightMem4::*;

import LSTM::*;
import Dense::*;

typedef enum {INIT, LOAD,INPUT} State deriving(Bits,Eq);
typedef enum {INIT, LSTM_1, LSTM_2, DENSE} Layer deriving(Bits,Eq);
typedef enum {INIT, KERNEL, RECURRENT, BIAS} Weight deriving(Bits,Eq);
typedef enum {INIT, INPUT, FORGET, CANDIDATE, OUTPUT, DENSE} Subweight deriving(Bits,Eq);
typedef enum {UPPER, LOWER} Mask deriving(Bits,Eq);

interface PredictiveMaintenanceIfc;
        method Action transmitWeight(Bit#(8) data);
		method Action transmitInput(Bit#(8) data);
        method ActionValue#(Bit#(16)) uartOut;
        method Bit#(3) rgbOut;
		method Bool loadStatus;
endinterface

module mkPredictiveMaintenance(PredictiveMaintenanceIfc);
		
		WeightMem4Ifc weightMem4 <- mkWeightMem4;
		
		Reg#(Bit#(14)) loadCounter <- mkReg(0);
		Reg#(Bit#(14)) addr <- mkReg(0);
		
		Reg#(State) predictMainState <- mkReg(LOAD);
		Reg#(Layer) layerState <- mkReg(LSTM_1);
		Reg#(Weight) weightState <- mkReg(KERNEL);
		Reg#(Subweight) subweightState <- mkReg(INPUT);
		Reg#(Mask) maskState <- mkReg(UPPER);
		Reg#(Bool) loadComplete <- mkReg(False);
		
		Bit#(16) lstm1KernelLen = 10000;
		Bit#(16) lstm1RecurrentLen = 40000;
		Bit#(16) lstm1BiasLen = 400;
		Bit#(16) lstm2KernelLen = 20000;
		Bit#(16) lstm2RecurrentLen = 10000;
		Bit#(16) lstm2BiasLen = 200;
		Bit#(16) denseKernelLen = 50;
		Bit#(16) denseBiasLen = 1;
		
		LSTMIfc#(10000, 40000, 400) lstm1 <- mkLSTM;
		LSTMIfc#(20000, 10000, 200) lstm2<- mkLSTM;
		//DenseIfc#(50) dense <- mkDense;
		//function Action updateLoad;
			
		//endfunction
		
		method Action transmitInput(Bit#(8) data);
			if (layerState == LSTM_1) begin
				lstm1.process(data);
			end
			else if (layerState == LSTM_2) begin
				lstm2.process(data);
			end
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
					weightMem4.writeWeight(pack(addr), maskeddata, 2'b00, bytemask);
				end
				FORGET: begin
					weightMem4.writeWeight(pack(addr), maskeddata, 2'b01, bytemask);
				end
				CANDIDATE: begin
					weightMem4.writeWeight(pack(addr), maskeddata, 2'b01, bytemask);
				end
				OUTPUT: begin
					weightMem4.writeWeight(pack(addr), maskeddata, 2'b11, bytemask);
				end
				INIT: begin
					weightMem4.readWeight(zeroExtend(data));
				end
			endcase
			
			Bit#(16) sectionLen = 0;
			case (layerState) matches
				LSTM_1: begin
					case (weightState) matches
						KERNEL: sectionLen = lstm1KernelLen/4;
						RECURRENT: sectionLen = lstm1RecurrentLen / 4;
						BIAS: sectionLen = lstm1BiasLen / 4;
					endcase
				end
				LSTM_2: begin
					case (weightState) matches
						KERNEL: sectionLen = lstm2KernelLen / 4;
						RECURRENT: sectionLen = lstm2RecurrentLen / 4;
						BIAS: sectionLen = lstm2BiasLen / 4;
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
								LSTM_1: layerUpdate = LSTM_2;
								LSTM_2: layerUpdate = DENSE;
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
			case (maskState) matches
				UPPER: maskState <= LOWER;
				LOWER: begin
					maskState <= UPPER;
					addr <= addr + 1;
				end
			endcase
        endmethod
		
		method ActionValue#(Bit#(16)) uartOut;
			let d <- weightMem4.resp;
			return zeroExtend(d);
		endmethod
		
		method Bool loadStatus;
			return loadComplete;
		endmethod
		
endmodule : mkPredictiveMaintenance
