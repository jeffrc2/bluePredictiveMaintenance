//Default packages
import FIFOF::*;
import BRAMCore::*;
import BRAMFIFO::*;
//up5k packages
import DSPArith::*;
//project packages

typedef enum {UPPER, LOWER} MaskHalf deriving (Bits,Eq);

typedef enum {INIT, 
	INPUT, 
	BIAS, FETCH} Stage deriving (Bits,Eq);//INPUT, HIDDEN, BIAS require weights from the SPRAM, while the ACTIVATEs do not.
	
typedef enum {INIT, MAIN} State deriving (Bits,Eq);

interface DenseIfc;
	method Action processWeight(Bit#(8) weight); //Put the weights into the weights queue.
	method Action processInput(Bit#(8) in); //Put the input into the input queue.
 
	method Bool inputReady;
	method Bool outputReady;
	method ActionValue#(Int#(8)) getOutput;
endinterface

module mkDense(DenseIfc);

	Reg#(Bool) run <- mkReg(False);

	FIFOF#(Int#(8)) inputQ <- mkSizedBRAMFIFOF(10);
	FIFOF#(Int#(8)) outputQ <- mkSizedBRAMFIFOF(1);
	
	rule start(inputQ.notEmpty);
		run <= True;
	endrule
	
	//weight storage
	BRAM_DUAL_PORT#(Bit#(9), Bit#(8)) bram0 <- mkBRAMCore2(51, False);
	
    Reg#(State) state <- mkReg(INIT);
	
//increment
	Reg#(Bit#(9)) bramAddrR <- mkReg(0);//Range 0-50
	FIFOF#(Stage) calcQ <- mkSizedBRAMFIFOF(2);
	FIFOF#(Stage) fetchQ <- mkSizedBRAMFIFOF(2);
	FIFOF#(Bit#(9)) bramAddrFetchQ <- mkSizedBRAMFIFOF(2);
	
	rule incrementMain(run && state == MAIN && bramAddrR < 50);
		bramAddrR <= bramAddrR + 1;
		bramAddrFetchQ.enq(bramAddrR);
		
		if (bramAddrR < 49) calcQ.enq(INPUT);
		else if (bramAddrR == 50) calcQ.enq(BIAS);
		fetchQ.enq(FETCH);
	endrule
	
	rule reset(run && state == MAIN && bramAddrR == 50);
		run <= False;
		state <= INIT;
		bramAddrR <= 0;
	endrule
	
	function Action bramReq(Bit#(9) bram_addr, BRAM_DUAL_PORT#(Bit#(9), Bit#(8)) bram_port);
		action
			bram_port.a.put(False, bram_addr, ?);
		endaction
	endfunction
	
	rule fetch(fetchQ.notEmpty);
		fetchQ.deq;
		Bit#(9) bramAddr = bramAddrFetchQ.first;
		bramAddrFetchQ.deq;
		bramReq(bramAddr, bram0);
	endrule
	
	function Int#(8) bramResp(BRAM_DUAL_PORT#(Bit#(9), Bit#(8)) bram_port);
		return unpack(bram_port.a.read);
	endfunction
	
	IntMult16x16Ifc dsp_mult <- mkIntMult16x16;

	Int#(16) invscale = 42;
	Int#(16) zeropoint = 1;
	
	function ActionValue#(Int#(8)) quantizedMult(Int#(8) x, Int#(8) y) 
		=
		actionvalue
			Int#(16) sx = signExtend(x);
			Int#(16) sy = signExtend(y);
			let p <- dsp_mult.calc(sx,sy);
			let s = truncate(p) / invscale;
			return truncate(s + zeropoint);
		endactionvalue;
		
	function Int#(8) quantizedAdd(Int#(8) x, Int#(8) y);
		Int#(16) sx = signExtend(x) - signExtend(zeropoint);
		Int#(16) sy = signExtend(y) - signExtend(zeropoint);
		let dx = sx / 2;
		let dy = sy / 2;
		let s = dx + dy;
		let s2 = s*2;
		return truncate(s2 + zeropoint);
	endfunction
	
	Reg#(Bool) bypass <- mkReg(False);
	
	function ActionValue#(Int#(8)) hardSigmoid(Int#(8) d)
		=
		actionvalue
			Int#(8) lowerlimit = -1;//-2.5f
			Int#(8) upperlimit = 1;//2.5f
			Int#(8) onepoint = 102; //1.0f
			Int#(8) alpha = 20; //0.2f
			Int#(8) offset = 51;//0.5f
			bypass <= !bypass;
			if (d <= lowerlimit) return truncate(zeropoint);
			else if (d >= upperlimit) return onepoint;
			else begin
				let m <- quantizedMult(alpha, d);
				return quantizedAdd(m, offset);
			end
		endactionvalue;
	
	Reg#(Int#(8)) summate <- mkReg(0);
	
	rule calcInput(calcQ.first == INPUT);
		Int#(8) coeffValue = bramResp(bram0);
		let inputValue = inputQ.first;
		inputQ.deq;
		let product <- quantizedMult(coeffValue, inputValue);
		Int#(8) aggregate = summate;
		summate <= quantizedAdd(aggregate, product);
	endrule
	
	Reg#(Bool) calcBias1 <- mkReg(False); //enables calcBiasZW
	
	rule calcBiasHS(calcQ.first == BIAS);
		let summate1 = summate;
		let summate2 <- hardSigmoid(summate1);
		summate <= summate;
		calcQ.deq;
		calcBias1 <= True;
	endrule
	
	rule calcBiasAdd(calcBias1);
		calcBias1 <= False;
		let biasValue = bramResp(bram0);
		outputQ.enq(quantizedAdd(biasValue, summate));
		summate <= 0;
	endrule

	function Int#(8) bramRespA(BRAM_DUAL_PORT#(Bit#(9), Bit#(8)) bram);
		return unpack(bram.a.read);
	endfunction

	//test code
	/*
	rule mainTest(state==MAIN);
		bram0.a.put(False, bramAddrW, ?);
	endrule
	
	rule mainResp(state==MAIN);
		Int#(8) out = bramRespA(bram0);
		outputQ.enq(out);
	endrule
	*/
	

	
//for weight processing
	FIFOF#(Bit#(8)) weightQ <- mkSizedBRAMFIFOF(1);
	Reg#(Bit#(9)) bramAddrW <- mkReg(0);//Range 0-50. Exclusively used by handle_weight
	Reg#(MaskHalf) writeMask <- mkReg(UPPER);//Exclusively used by handle_weight
	
	
	
//weight processing
	method Action processWeight(Bit#(8) weight);
        if (bramAddrW < 50) begin 
                    bramAddrW <= bramAddrW + 1;
		end else begin //enable main
                    bramAddrW <= 0;
                    state <= MAIN;
        end
		bram0.b.put(True, bramAddrW, weight);
		$display("Writing Dense weight bramAddr ", bramAddrW, " data ", weight);
    endmethod
	

//input processing
	method Action processInput(Bit#(8) weight);
		inputQ.enq(unpack(weight));
    endmethod
	
	method Bool inputReady;//ready to queue input
		return inputQ.notFull;
	endmethod
	
	method Bool outputReady;//ready to dequeue output
		return outputQ.notEmpty;
	endmethod

	method ActionValue#(Int#(8)) getOutput;
		outputQ.deq;
		return outputQ.first;
	endmethod

endmodule: mkDense
