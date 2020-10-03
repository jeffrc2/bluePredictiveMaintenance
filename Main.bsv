import FIFO::*;
import BRAM::*;
import BRAMFIFO::*;

import LSTM1::*;

typedef enum {LSTM1, LSTM2, DENSE, INPUT, FIN} LoadStage deriving (Bits,Eq);

interface MainIfc;
	method Action uartIn(Bit#(8) data);
	method ActionValue#(Bit#(8)) uartOut;
	method Bit#(3) rgbOut;
endinterface


module mkMain(MainIfc);
	Clock curclk <- exposeCurrentClock;
	
	LSTM1Ifc lstm1 <- mkLSTM1;
	
	//PredictiveMaintenance processes input and weights
	
	FIFO#(Bit#(8)) uartQ <- mkSizedBRAMFIFO(2);
	
	Reg#(int) counter <- mkReg(0);
	
	rule count;
		counter <= counter + 1;
		$display("top counter", counter);

	endrule
	
	rule relayUart;
		Int#(8) out <- lstm1.getOutput;
		uartQ.enq(pack(out));
	endrule
	
	Reg#(int) weight_counter <- mkReg(0);
	Reg#(LoadStage) loadStage <- mkReg(LSTM1);
	
	
	//Method transfers data to LSTM1 to be processed as weights until all weights have been processed, then subsequent data is transferred to be processed as input
	method Action uartIn(Bit#(8) data);
		$display("uartIn running");
		case (loadStage) matches
			LSTM1: begin
				lstm1.processWeight(data);
				if (weight_counter < 50399) weight_counter <= weight_counter + 1;
				else begin
					weight_counter <= 0;
					loadStage <= LSTM2;
					$display("LSTM1 weights loaded.");
				end
			end
			LSTM2: begin
				lstm1.processLSTM2Weight(data);
				if (weight_counter < 30199) weight_counter <= weight_counter + 1;
				else begin
					weight_counter <= 0;
					loadStage <= DENSE;
					$display("LSTM2 weights loaded.");
				end
			end
			DENSE: begin
				lstm1.processDenseWeight(data);
				if (weight_counter < 50) weight_counter <= weight_counter + 1;
				else begin
					weight_counter <= 0;
					loadStage <= INPUT;
					$display("Dense weights loaded.");
				end
			end
			/*
			INPUT: begin
				lstm1.processInput(data);
				if (weight_counter < 1249) weight_counter <= weight_counter + 1;
				else begin
					weight_counter <= 0;
					loadStage <= FIN;
				end
				
			end
			*/
		endcase
	endmethod
	
	method ActionValue#(Bit#(8)) uartOut;
		uartQ.deq;
		$display("output given");
		$finish(0);
		return uartQ.first;
		
	endmethod
	
	method Bit#(3) rgbOut;
		//0 when idle
		//1 when receiving weights
		//2 when input is being received
		//3 when LSTM1 is running
		//4 when LSMT2 is running
		//5 when Dense is running
		//6 when PredictiveMaintenance is done.
		return 0;
	endmethod
endmodule