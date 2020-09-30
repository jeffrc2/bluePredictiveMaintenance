
import FIFOF::*;
import FIFO::*;
import BRAMCore::*;
import BRAMFIFO::*;

typedef enum {INIT, 
	INPUT,  
	BIAS} Stage deriving(Bits,Eq);


interface DenseIfc;
	method Action processInput(Int#(8) in);
	method Action processWeight(Bit#(8) weights);
	method Action start;
	method Bool inputReady;

	method Bool outputReady;
	method ActionValue#(Int#(8)) getOutput;
endinterface

module mkDense(DenseIfc);
	
	FIFOF#(Int#(8)) inputQ <- mkSizedBRAMFIFOF(50);
	FIFOF#(Int#(8)) outputQ <- mkSizedBRAMFIFOF(2);
	
	Int#(16) invscale = 102;
	Int#(16) zeropoint = 0;
	
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
	
	BRAM_DUAL_PORT#(Bit#(9), Bit#(8)) bram_weights <- mkBRAMCore2(51, False);
	
	function Int#(8) bramRespA(BRAM_DUAL_PORT#(Bit#(9), Bit#(8)) bram);
		return unpack(bram.a.read);
	endfunction
	
	Reg#(Bit#(9)) mainIteration <- mkReg(0);
	Reg#(Bit#(9)) bramAddr <- mkReg(0);//Range 0-50
	
	Reg#(Stage) fetchStage <- mkReg(INIT);
	//FIFO#(Stage) fetchStageQ <- mkFIFO;
	Reg#(Stage) calcStage <- mkReg(INIT);
	
	Reg#(Bool) mainIncrementEN <- mkReg(False);
	
	rule cascadeFetchCalc;
		calcStage <= fetchStage;
		//fetchStageQ.deq;
		//calcStage <= fetchStageQ.first;
		//$display( "going to %d", pack(fetchStage) );
	endrule
	
	rule incrementMain(mainIncrementEN && mainIteration < 50);
		fetchStage <= INPUT;
		bramAddr <= mainIteration;
		mainIteration <= mainIteration + 1;
	endrule
	
	rule incrementBias(mainIncrementEN && mainIteration == 50);
		fetchStage <= BIAS;
		mainIteration <= mainIteration + 1;
	endrule
	
	rule incrementReset(mainIncrementEN && mainIteration == 51);
		mainIncrementEN <= False;
		fetchStage <= INIT;
	endrule
	
	rule fetch(fetchStage == INPUT || fetchStage == BIAS);
		`ifdef BSIM
			$display("dense fetch");
			$display("iteration ", mainIteration);
		`endif
		bram_weights.a.put(False, bramAddr, ?);
	endrule
	
	Reg#(Int#(8)) summate <- mkReg(0);
	Reg#(Int#(8)) bias <- mkReg(0);
	
	Reg#(Bool) calcBias1 <- mkReg(False); //enables calcBiasZW
	
	rule calcBiasCascade(calcStage == BIAS);//follows fetchStage in a cascaded delay 
		calcBias1 <= True;
	endrule
	
	rule calcInput(calcStage == INPUT);
		`ifdef BSIM
			$display("dense input");
		`endif
		Int#(8) coeff = bramRespA(bram_weights); 
		inputQ.deq;
		Int#(8) dataIn = inputQ.first;
		Int#(8) product = quantizedMult(coeff, dataIn);
		Int#(8) aggregate = summate;
		summate <= quantizedAdd(aggregate, product);
	endrule

	rule calcBiasHS(calcStage == BIAS);
		`ifdef BSIM
			$display("dense bias");
		`endif
		summate <= hardSigmoid(summate);
		bias <= bramRespA(bram_weights);
	endrule	
	
	rule calcBiasAdd(calcBias1);
		//Int#(8) bias = bramRespA(bram_weights);
		outputQ.enq(quantizedAdd(bias, summate));
		`ifdef BSIM
			$display("Finished process");
			//$finish(0);
		`endif
	endrule
	
	Reg#(Bit#(9)) bramWriteAddr <- mkReg(0);
	
	method Action processWeight(Bit#(8) weight);
		`ifdef BSIM
			$display("processing Dense weight");
		`endif
		bram_weights.b.put(True, bramWriteAddr, weight);
		if (bramWriteAddr < 50) bramWriteAddr <= bramWriteAddr + 1;
		else bramWriteAddr <= 0;
	endmethod
	
	method Action processInput(Int#(8) in);
		`ifdef BSIM
			$display("processing Dense input");
		`endif
		inputQ.enq(in);
		
	endmethod
		
	method Action start;
		`ifdef BSIM
			$display("dense start");
			//$finish(0);
		`endif
		if (fetchStage == INIT) begin
			fetchStage <= INPUT;
			//fetchStageQ.enq(INPUT);
			mainIncrementEN <= True;
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