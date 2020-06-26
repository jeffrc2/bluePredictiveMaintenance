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
	
	method Action uartIn(Bit#(8) data);
		//lstm_1_kernel 10000
		//lstm_1_recurrent 40000
		//lstm_1_bias 400 
		//lstm_2_kernel 20000
		//lstm_2_recurrent 10000
		//lstm_2_bias 200
		//dense_kernel 50
		//dense_bias 1
		if (!predictiveMaintenance.loadStatus) begin
			predictiveMaintenance.transmitWeight(data);
		end
		else begin
			predictiveMaintenance.transmitInput(data);
		end
	endmethod
	method ActionValue#(Bit#(8)) uartOut;
		let d <- predictiveMaintenance.uartOut;
		return truncate(d);
	endmethod
	method Bit#(3) rgbOut;
		//0 when idle
		//1 when receiving weights
		//2 when processing
		//3 when done.
		return 0;
	endmethod
endmodule
