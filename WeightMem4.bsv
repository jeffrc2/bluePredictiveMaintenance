import Spram::*;
import FIFO::*;
import BRAM::*;
import BRAMFIFO::*;


typedef enum {WRITE, READ} State deriving(Bits,Eq);

interface WeightMem4Ifc;
	method Action writeWeight(Bit#(14) addr, Bit#(16) data, Bit#(2) ramID, Bit#(4) bytemask); //Write a weight into half of the space depending on the bytemask
	method Action reqWeight(Bit#(14) addr, Bit#(2) netType); //Put a weight request into the SPRAM, with an accompanying network ID
	method ActionValue#(Tuple2#(Bit#(64), Bit#(2))) recvWeight; //Retrieve a weight, along with an accompanying network ID
endinterface

module mkWeightMem4(WeightMem4Ifc);
	Clock curclk <- exposeCurrentClock;

	//SPRAMs for corresponding I,F,C,O weights
	Spram256KAIfc ramI <- mkSpram256KA;
	Spram256KAIfc ramF <- mkSpram256KA;
	Spram256KAIfc ramC <- mkSpram256KA;
	Spram256KAIfc ramO <- mkSpram256KA;
	
	//relay queue holds onto the network ID
	FIFO#(Bit#(2)) relayQ <- mkFIFO;
	
	//output queue holds requested weight data until released by reqWeight
	FIFO#(Tuple2#(Bit#(64), Bit#(2))) outputQ <- mkFIFO;
	
	//readReady
	//Reg#(Bool) readReady <- mkReg(False);
	
	//Reg#(Bit#(16)) d <- mkReg(0);

	//Rule retrieves all SPRAM outputs and wraps them and associated network for retrieval.
	rule transfer;//(readReady == True);
		
		let i <- ramI.resp;
		let f <- ramF.resp;
		let c <- ramC.resp;
		let o <- ramO.resp;
		//d <= i;
		Bit#(64) set;
		set[63:48] = i;
		set[47:32] = f;
		set[31:16] = c;
		set[15:0] = o;
	
		Tuple2#(Bit#(64), Bit#(2)) weights = tuple2(set, relayQ.first);
		relayQ.deq;
		//`ifdef BSIM
		//$display("transfer %u %u %u %u %u %u", i, f, c, o, relayQ.first);
		//`endif
		outputQ.enq(weights);
		//readReady <= False;
	endrule
	
	//Method sends request address to all 4 SPRAMs and the network ID for retrieval
	method Action reqWeight(Bit#(14) addr, Bit#(2) netType);
		ramI.req(addr, ?, False, ?);
		ramF.req(addr, ?, False, ?);
		ramC.req(addr, ?, False, ?);
		ramO.req(addr, ?, False, ?);
		relayQ.enq(netType);
		//`ifdef BSIM
		//$display("reqWeight %u", addr);
		//`endif
		//readReady <= True;
	endmethod
	
	//Retrieve a completed weight request's weights and network ID
	method ActionValue#(Tuple2#(Bit#(64), Bit#(2))) recvWeight;
		outputQ.deq;
		//`ifdef BSIM
		//$display("recv check");
		//`endif
		return outputQ.first;
		
	endmethod
	
	//Writes a weight into 1 of 4 SPRAM modules based on the provided ramID
	method Action writeWeight(Bit#(14) addr, Bit#(16) data, Bit#(2) ramID, Bit#(4) bytemask);
		//`ifdef BSIM
		//$display("writeWeight addr: %u ramID: %u data: %u, bytemask: %u", addr, ramID, data, bytemask);
		//`endif
		case (ramID) matches
			2'b00: ramI.req(addr, data, True, bytemask);
			2'b01: ramF.req(addr, data, True, bytemask);
			2'b10: ramC.req(addr, data, True, bytemask);
			2'b11: ramO.req(addr, data, True, bytemask);
		endcase
	endmethod
	
	
endmodule : mkWeightMem4