
import FIFOF::*;

import BRAMFIFO::*;

typedef enum {INIT, KERNEL, BIAS, FIN} Stage deriving(Bits,Eq);

typedef enum {CALC1, CALC2} CalcStage deriving(Bits,Eq);

typedef enum {INIT, CALC, FIN} State deriving(Bits,Eq);

interface DenseIfc#(numeric type height, numeric type bias, numeric type offset);
	method ActionValue#(Bit#(14)) deqBRAMReq;
	method Action processInput(Int#(8) in);
	method Action processWeight(Bit#(64) weights);
	method Action start;
	method ActionValue#(Bit#(14)) getRequest;
	method Bool inputReady;
	method Bool requestQueued;
	method Bool outputQueued;
	method ActionValue#(Int#(8)) getOutput;
endinterface

module mkDense(DenseIfc#(height, bias, offset));
	Reg#(Bit#(14)) height_counter <- mkReg(0);//50
	Reg#(Bit#(8)) bias_counter <- mkReg(0);
	
	Int#(16) invscale = 102;
	Int#(16) zeropoint = 0;

	Reg#(State) state <- mkReg(INIT);
	
	
	Reg#(Bit#(8)) dense_height_counter <- mkReg(0);//50
	Reg#(Bit#(8)) dense_bias_counter <- mkReg(0);
	
	FIFOF#(Int#(8)) inputQ <- mkSizedBRAMFIFOF(2);
	FIFOF#(Bit#(14)) reqQ <- mkSizedBRAMFIFOF(2);
	FIFOF#(Bit#(64)) spramQ <- mkSizedBRAMFIFOF(2);
	FIFOF#(Int#(8)) outputQ <- mkSizedBRAMFIFOF(2);
	
	Reg#(Stage) denseStage <- mkReg(INIT);
	
	Reg#(CalcStage) calcStage <- mkReg(CALC1);
	
	Reg#(Stage) reqStage <- mkReg(INIT);
	
	Reg#(Int#(8)) accum1 <- mkReg(0);
	
	Reg#(Int#(8)) accum2 <- mkReg(0);
	
	rule reqRAM(reqStage != INIT && reqStage != FIN && reqQ.notFull);
		Bit#(14) offset = fromInteger(valueOf(offset));
		Bit#(14) height = fromInteger(valueOf(height));
		if (reqStage == KERNEL) begin
			Bit#(14) req_addr = offset + height_counter;
			reqQ.enq(req_addr);
			if (height_counter < height - 1)
				height_counter <= height_counter + 1;//extracting two at a time.
			else begin 
				height_counter <= 0;
				reqStage <= BIAS;
			end
		end
		else if (reqStage == BIAS) begin
			Bit#(14) req_addr = offset + height + 1;
			reqStage <= FIN;
		end
	endrule	
	
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
	
	
	rule summate1(state == CALC && calcStage == CALC1 && denseStage != FIN && spramQ.notEmpty);
		`ifdef BSIM
		$display("summate1");
		`endif
		Bit#(64) weights = spramQ.first;
		Int#(8) wt = unpack(weights[15:8]);
		spramQ.deq;
		case (denseStage) matches
			KERNEL: begin
				Int#(8) dataIn = inputQ.first;	
				inputQ.deq;
				
				Int#(8) mult = quantizedMult(dataIn, wt);
				accum1 <= mult;
				calcStage <= CALC2;
				
				if (dense_height_counter == fromInteger(valueOf(height)) - 1) begin
					denseStage <= BIAS;
					height_counter <= 0;
				end else dense_height_counter <= dense_height_counter + 1;
			end
			BIAS: begin
				Int#(8) processed = quantizedAdd(accum2, wt);
				accum2 <= processed;
			end
		endcase
	endrule
	
	rule summate2(calcStage == CALC2);
		`ifdef BSIM
		$display("summate2");
		`endif
		case (denseStage) matches
			KERNEL: begin
				accum2 <= quantizedAdd(accum2, accum1);
				calcStage <= CALC1;
			end
			BIAS: begin
				Int#(8) sigmoid = accum2;
				outputQ.enq(sigmoid);
				denseStage <= FIN;
				state <= FIN;
				$finish(1);
			end
		endcase
		
	endrule
		
	
	
	method Action processWeight(Bit#(64) weights);
		spramQ.enq(weights);
	endmethod
	
	method Action processInput(Int#(8) in);
		inputQ.enq(in);
	endmethod
		
	method Action start;
		if (reqStage == INIT) begin
			denseStage <= KERNEL;
			reqStage <= KERNEL;
			state <= CALC;
			`ifdef BSIM
			$display("dense start");
			`endif
		end
	endmethod 
	
	method ActionValue#(Bit#(14)) getRequest;
		reqQ.deq;
		return reqQ.first;
	endmethod
	
	method Bool inputReady;//ready to queue input
		return !inputQ.notEmpty;
	endmethod
	
	method Bool requestQueued;//if a memory read request is Queued
		return reqQ.notEmpty;
	endmethod
	
	method Bool outputQueued;
		return outputQ.notEmpty;
	endmethod
	
	method ActionValue#(Int#(8)) getOutput;
		outputQ.deq;
		return outputQ.first;
	endmethod
	

endmodule: mkDense