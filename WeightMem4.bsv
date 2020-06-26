import Spram::*;
import FIFO::*;
import BRAM::*;
import BRAMFIFO::*;

interface WeightMem4Ifc;
	method Action writeWeight(Bit#(14) addr, Bit#(16) data, Bit#(2) ramID, Bit#(4) bytemask);
	method Action readWeight(Bit#(14) addr);
	method ActionValue#(Bit#(8)) resp;
	
	
endinterface

module mkWeightMem4(WeightMem4Ifc);
	Clock curclk <- exposeCurrentClock;

	Spram256KAIfc ramI <- mkSpram256KA;
	Spram256KAIfc ramF <- mkSpram256KA;
	Spram256KAIfc ramC <- mkSpram256KA;
	Spram256KAIfc ramO <- mkSpram256KA;
	
	FIFO#(Bit#(8)) relayUart <- mkFIFO;
	
	rule ttt;
		let d <- ramI.resp;
		let e <- ramF.resp;
		let f <- ramC.resp;
		let g <- ramO.resp;
		d = d - e;
		f = f - g;
		let h = d - f;
		relayUart.enq(truncate(h));
	endrule
	
	method Action readWeight(Bit#(14) addr);
		ramI.req(addr, ?, True, 4'b1111);
		ramF.req(addr, ?, True, 4'b1111);
		ramC.req(addr, ?, True, 4'b1111);
		ramO.req(addr, ?, True, 4'b1111);
	endmethod
	
	method Action writeWeight(Bit#(14) addr, Bit#(16) data, Bit#(2) ramID, Bit#(4) bytemask);
		case (ramID) matches
			2'b00: ramI.req(addr, data, True, bytemask);
			2'b01: ramF.req(addr, data, True, bytemask);
			2'b10: ramC.req(addr, data, True, bytemask);
			2'b11: ramO.req(addr, data, True, bytemask);
		endcase
	endmethod
	
	method ActionValue#(Bit#(8)) resp;
		relayUart.deq;
		let d = relayUart.first;
		return d;
	endmethod
	
endmodule : mkWeightMem4