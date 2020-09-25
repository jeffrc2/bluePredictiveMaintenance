
import FIFOF::*;
import BRAMCore::*;
import BRAMFIFO::*;

typedef enum {INIT, FETCH, INCREMENT, INPUT_MULT, INPUT_ADD, BIAS_HS, BIAS_ADD, FIN} CalcStage deriving(Bits,Eq);


interface DenseIfc;
	method Action processInput(Int#(8) in);
	method Action processWeight(Bit#(8) weights);
	method Action start;
	method Bool inputReady;

	method Bool outputReady;
	method ActionValue#(Int#(8)) getOutput;
endinterface

module mkDense(DenseIfc);
	
	Int#(16) invscale = 102;
	Int#(16) zeropoint = 0;
	
	FIFOF#(Int#(8)) inputQ <- mkSizedBRAMFIFOF(2);
	FIFOF#(Int#(8)) outputQ <- mkSizedBRAMFIFOF(2);
	
	
	Reg#(CalcStage) calcStage <- mkReg(INIT);
	
	BRAM_DUAL_PORT#(Bit#(9), Bit#(8)) bram_weights <- mkBRAMCore2(51, False);
	
	function Int#(8) bramRespA(BRAM_DUAL_PORT#(Bit#(9), Bit#(8)) bram);
		return unpack(bram.a.read);
	endfunction
	
	Reg#(Bit#(9)) iteration <- mkReg(0);
	
	rule fetch(calcStage == FETCH);
		`ifdef BSIM
			//$display("dense fetch");
		`endif
		bram_weights.a.put(False, iteration, ?);
		if (iteration < 50) calcStage <= BIAS_HS;
		else calcStage <= FIN;
	endrule
	
	rule increment(calcStage == INCREMENT);
		`ifdef BSIM
			$display("dense increment");
		`endif
		iteration <= iteration + 1;
		calcStage <= FETCH;
	endrule
	

	
	Reg#(Int#(8)) temp <- mkReg(0);
	Reg#(Int#(8)) summate <- mkReg(0);
	
	function Int#(8) quantizedMult(Int#(8) x, Int#(8) y);

		Int#(16) sx = signExtend(x);
		Int#(16) sy = signExtend(y);
		let p = sx * sy;
		let s = p / invscale;
		return truncate(s + zeropoint);
	endfunction
	
	function Int#(8) quantizedAdd(Int#(8) x, Int#(8) y);
		Int#(16) sx = signExtend(x) - signExtend(zeropoint);
		Int#(16) sy = signExtend(y) - signExtend(zeropoint);
		let dx = sx / 2;
		let dy = sy / 2;
		let s = dx + dy;
		let s2 = s*2;
		return truncate(s2 + zeropoint);
	endfunction
		
	function Int#(8) hardSigmoid(Int#(8) d);
		Int#(8) lowerlimit = -1;//-2.5f
		Int#(8) upperlimit = 1;//2.5f
		Int#(8) onepoint = 102; //1.0f
		Int#(8) alpha = 20; //0.2f
		Int#(8) offset = 51;//0.5f
		if (d <= lowerlimit) return truncate(zeropoint);
		else if (d >= upperlimit) return onepoint;
		else return quantizedAdd(quantizedMult(alpha, d), offset);
	endfunction	
	
	rule input_QMult(calcStage == INPUT_MULT);
		`ifdef BSIM
			$display("dense input");
		`endif
		Int#(8) coeff = bramRespA(bram_weights); 
		Int#(8) dataIn = inputQ.first;
		inputQ.deq;
		temp <= quantizedMult(coeff, dataIn);
		calcStage <= INPUT_ADD;
	endrule
	
	rule input_QAdd(calcStage == INPUT_ADD);

		Int#(8) product = temp;
		Int#(8) aggregate = summate;
		temp <= quantizedAdd(aggregate, product);
		calcStage <= INCREMENT;
	endrule
	
	rule bias_HS(calcStage == BIAS_HS);
	`ifdef BSIM
			$display("dense bias");
		`endif
		summate <= hardSigmoid(summate);
		calcStage <= BIAS_ADD;
	endrule	
	
	rule bias_QAdd(calcStage == BIAS_ADD);
		Int#(8) bias = bramRespA(bram_weights);
		outputQ.enq(quantizedAdd(bias, summate));
		calcStage <= FIN;
	endrule
	
	Reg#(Bit#(9)) bramWriteAddr <- mkReg(0);
	
	method Action processWeight(Bit#(8) weight);
		`ifdef BSIM
			$display("processing Dense weight");
		`endif
		bram_weights.b.put(True, bramWriteAddr, weight);
		bramWriteAddr <= bramWriteAddr + 1;
	endmethod
	
	method Action processInput(Int#(8) in);
		inputQ.enq(in);
	endmethod
	
		
	method Action start;
		if (calcStage == INIT) begin
			calcStage <= FETCH;
			bram_weights.a.put(False, iteration, ?);
			`ifdef BSIM
				$display("dense start");
			`endif
		end
	endmethod 
	
	method Bool inputReady;//ready to queue input
		return !inputQ.notEmpty;
	endmethod
	
	method Bool outputReady;//ready to dequeue output
		return outputQ.notEmpty;
	endmethod

	
	method ActionValue#(Int#(8)) getOutput;
		outputQ.deq;
		return outputQ.first;
	endmethod
	

endmodule: mkDense