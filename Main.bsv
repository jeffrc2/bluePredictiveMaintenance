import FIFO::*;
import BRAM::*;
import BRAMFIFO::*;

import PredictiveMaintenance::*;
import Spram::*;

interface MainIfc;
	method Action uartIn(Bit#(8) data);
	method ActionValue#(Bit#(8)) uartOut;
	method Bit#(3) rgbOut;
endinterface



module mkMain(MainIfc);
	Clock curclk <- exposeCurrentClock;

	PredictiveMaintenanceIfc predictiveMaintenance <- mkPredictiveMaintenance;
	
	FIFO#(Bit#(8)) uartQ <- mkSizedFIFO(2);
	
	Reg#(int) counter <- mkReg(0);
	
	rule count;
		counter <= counter + 1;
	endrule
	
	rule relayUart(predictiveMaintenance.uartOutReady);
		Bit#(8) out <- predictiveMaintenance.uartOut;
		uartQ.enq(out);
	endrule
	
	method ActionValue#(Bit#(8)) uartOut;
		uartQ.deq;
		return uartQ.first;
	endmethod
	
	method Action uartIn(Bit#(8) data) if (predictiveMaintenance.uartStatus == True);
		if (predictiveMaintenance.loadStatus == False) begin
			predictiveMaintenance.transmitWeight(data);
		end
		else begin
			predictiveMaintenance.transmitInput(data);
		end
	endmethod
	
	method Bit#(3) rgbOut;
		//0 when idle
		//1 when receiving weights
		//2 when processing
		//3 when done.
		return 0;
	endmethod
endmodule
