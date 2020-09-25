import BRAMCore::*;
import FIFOF::*;
import BRAMFIFO::*;
import LSTM2::*;
import Spram::*;
//invscale = 42;
//zeropoint = 0;

//lowerlimit = -1;//-2.5f
//upperlimit = 1;//2.5f
//onepoint = 102; //1.0f
//alpha = 20; //0.2f
//offset = 51;//0.5f

typedef enum {UPPER, LOWER} MaskHalf deriving (Bits,Eq);

typedef enum {INIT, 
	INCREMENT, FETCH, 
	INPUT_MULT, INPUT_ADD, 
	HIDDEN_MULT, HIDDEN_ADD, 
	BIAS1_ADD1, BIAS1_ADD2, BIAS1_HS,
	BIAS2_Mult,
	BIAS3_Mult,
	BIAS4_ADD,
	BIAS5_HS,
	BIAS6_Mult,
	FIN} CalcStage deriving (Bits,Eq);

interface LSTM1Ifc;
	method Action processInput(Bit#(8) in); //Put the input into the input queue.
	method Action processWeight(Bit#(8) weight); //Put the weights into the weights queue.
	method Action processLSTM2Weight(Bit#(8) weight);
	method Action processDenseWeight(Bit#(8) weight); 
	method Action start; 
	method Bool inputReady;
	method Bool outputReady;
	method ActionValue#(Int#(8)) getOutput;
endinterface

module mkLSTM1(LSTM1Ifc);

	LSTM2Ifc lstm2 <- mkLSTM2; 
	
	Reg#(CalcStage) calcStage <- mkReg(INIT);

	Reg#(MaskHalf) mask <- mkReg(UPPER);
	
	//Reg#(Bit#(11)) input_counter <- mkReg(0);

	FIFOF#(Int#(8)) inputQ <- mkSizedBRAMFIFOF(10);
	FIFOF#(Int#(8)) outputQ <- mkSizedBRAMFIFOF(2);
	
	Int#(16) invscale = 42;
	Int#(16) zeropoint = 1;
	
	BRAM_DUAL_PORT#(Bit#(9), Bit#(8)) bram_hidden <- mkBRAMCore2(100, False);
	BRAM_DUAL_PORT#(Bit#(9), Bit#(8)) bram_carry <- mkBRAMCore2(100, False);
	
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
	
	Spram256KAIfc spram0 <- mkSpram256KA;
	Spram256KAIfc spram1 <- mkSpram256KA;
	
	Reg#(Bit#(15)) spramAddr <- mkReg(0);

	Reg#(Int#(8)) temp <- mkReg(0);
	
	BRAM_DUAL_PORT#(Bit#(9), Bit#(8)) bram_x <- mkBRAMCore2(512, False);
	BRAM_DUAL_PORT#(Bit#(9), Bit#(8)) bram_y <- mkBRAMCore2(512, False);
	
	Reg#(Bool) spramRead <- mkReg(False);
	Reg#(Bool) bramRead <- mkReg(False);
	
	function Int#(8) spramSelect(Bit#(16) spramValuePair);
		
		Int#(8) value = 0;
		case (mask) matches
			UPPER: value = unpack(spramValuePair[15:8]);
			LOWER: value = unpack(spramValuePair[7:0]);
		endcase
		return value;
	endfunction
	
	
	function Int#(8) bramRespA(BRAM_DUAL_PORT#(Bit#(9), Bit#(8)) bram);
		return unpack(bram.a.read);
	endfunction
	
	function Int#(8) bramRespB(BRAM_DUAL_PORT#(Bit#(9), Bit#(8)) bram);
		return unpack(bram.b.read);
	endfunction
	
	rule relay(lstm2.outputReady);
		Int#(8) dataOut <- lstm2.getOutput;
		outputQ.enq(dataOut);
	endrule
	
	Reg#(Bit#(16)) iteration <- mkReg(0);
	Reg#(Bit#(9)) access_iteration <- mkReg(0);
	Reg#(Bit#(9)) unit_iteration <- mkReg(0);
	Reg#(Bit#(9)) layer_iteration <- mkReg(0);
	Reg#(Bit#(9)) writeAddr <- mkReg(0);
	
	rule input_QMult(calcStage == INPUT_MULT); //100x4 steps before incrementing 
		`ifdef BSIM
			$display("lstm1 input");
		`endif
		Bit#(16) valuepair <- spram0.resp;
		Int#(8) coeff = spramSelect(valuepair);
		Int#(8) dataIn = inputQ.first;
		temp <= quantizedMult(coeff, dataIn);
		calcStage <= INPUT_ADD;
	endrule
	
	rule input_QAdd(calcStage == INPUT_ADD);
		
		Int#(8) aggregate = 0;
		if  (iteration > 399) begin
			aggregate = bramRespA(bram_x);
		end
		Int#(8) product = temp;
		bram_x.b.put(True, writeAddr, pack(quantizedAdd(aggregate, product)));
		calcStage <= INCREMENT;
	endrule
	
	rule hidden_QMult(calcStage == HIDDEN_MULT);
		`ifdef BSIM
			$display("lstm1 hidden");
		`endif
		Bit#(16) valuepair = 0; 
		case (spramAddr[14]) matches
			0: valuepair <- spram0.resp;
			1: valuepair <- spram1.resp;
		endcase
		Int#(8) coeff = spramSelect(valuepair);
		Int#(8) hiddenIn = 0;
		if (access_iteration != 0) begin
			hiddenIn = bramRespA(bram_hidden);
		end
		temp <= quantizedMult(coeff, hiddenIn);
		calcStage <= HIDDEN_ADD;
	endrule
	
	rule hidden_QAdd(calcStage == HIDDEN_ADD);
		Int#(8) aggregate = 0;
		if (iteration > 10399) begin
			aggregate = bramRespA(bram_y);
		end
		Int#(8) product = temp;
		bram_x.b.put(True, writeAddr, pack(quantizedAdd(aggregate, product)));
		calcStage <= INCREMENT;
	endrule
	
	rule bias1_QAdd1(calcStage == BIAS1_ADD1);
		`ifdef BSIM
			$display("lstm1 bias1");
		`endif
		Int#(8) x = bramRespA(bram_x);
		Int#(8) y = bramRespA(bram_y);
		temp <= quantizedAdd(x, y);
		calcStage <= BIAS1_ADD2;
	endrule
	
	rule bias1_QAdd2(calcStage == BIAS1_ADD2);
		Bit#(16) valuepair <- spram1.resp;
		Int#(8) bias = spramSelect(valuepair);
		Int#(8) sum = temp;
		temp <= quantizedAdd(bias, sum);
		calcStage <= BIAS1_HS;
	endrule
	
	rule bias1_HS(calcStage == BIAS1_HS);
		Int#(8) sum = temp;
		
		bram_y.b.put(True, writeAddr, pack(hardSigmoid(sum)));
		calcStage <= INCREMENT;
	endrule
	
	rule bias2_CMult(calcStage == BIAS2_Mult);
		`ifdef BSIM
			$display("lstm1 bias2");
		`endif
		Int#(8) c_prev = bramRespA(bram_carry);
		Int#(8) f = bramRespA(bram_y);
		bram_carry.b.put(True, writeAddr, pack(quantizedMult(c_prev,f)));
		calcStage <= INCREMENT;
	endrule
	
	rule bias3_QMult(calcStage == BIAS3_Mult);
		`ifdef BSIM
			$display("lstm1 bias3");
		`endif
		Int#(8) i = bramRespA(bram_y);
		Int#(8) c = bramRespB(bram_y);
		bram_hidden.b.put(True, writeAddr, pack(quantizedMult(i,c)));
		calcStage <= INCREMENT;
	endrule
	
	rule bias4_QAdd(calcStage == BIAS4_ADD);
		`ifdef BSIM
			$display("lstm1 bias4");
		`endif
		Int#(8) cf = bramRespA(bram_carry);
		Int#(8) ic = bramRespA(bram_hidden);
		bram_carry.b.put(True, writeAddr, pack(quantizedAdd(cf,ic))); //new carry state
		calcStage <= INCREMENT;
	endrule
	
	rule bias5_HS(calcStage == BIAS5_HS);
		`ifdef BSIM
			$display("lstm1 bias5");
		`endif
		Int#(8) c_state = bramRespA(bram_carry);
		bram_hidden.b.put(True, writeAddr, pack(hardSigmoid(c_state)));
		calcStage <= INCREMENT;
	endrule
	
	Reg#(Bit#(6)) height_count <- mkReg(0);
	
	rule bias6_QMult(calcStage == BIAS6_Mult && lstm2.inputReady); //must wait for LSTM2 queue to be empty before allowed to continue.
		`ifdef BSIM
			$display("lstm1 bias6");
		`endif
		Int#(8) hs = bramRespA(bram_hidden);
		Int#(8) o = bramRespA(bram_y);
		let h_state = quantizedMult(hs, o);
		bram_hidden.b.put(True, writeAddr, pack(h_state)); //new hidden state
		lstm2.processInput(h_state);
		lstm2.start;
		calcStage <= INCREMENT;
	endrule 
	
	rule bias6_QMult_full(calcStage == BIAS6_Mult && !lstm2.inputReady);
		//not ready for input yet.
		`ifdef BSIM
			$display("lstm1 waiting");
		`endif
	endrule
	

	
	rule fetcher(calcStage == FETCH); //send fetch requests; 
		`ifdef BSIM
			//$display("lstm1 fetch");
		`endif
		if (iteration < 10000 ) begin //input calculation requests
			if (access_iteration == 399) begin //dequeue input
				inputQ.deq;
			end
			
			Bit#(9) bramAddr1 = access_iteration;
			writeAddr <= unit_iteration;
			
			spram0.req(spramAddr[13:0], ?, False, ?);
			bram_x.a.put(False, bramAddr1, ?);
			calcStage <= INPUT_MULT;
		end
		else if (iteration >= 10000 && iteration < 50000) begin //hidden calculation requests
			if (access_iteration == 399) begin //request the next hidden state
				Bit#(9) bramAddr2 = access_iteration; //possibly incorrect
				bram_hidden.a.put(False, bramAddr2, ?);
			end
			
			Bit#(9) bramAddr1 = access_iteration;
			writeAddr <= unit_iteration;
			case (spramAddr[14]) matches
				0: spram0.req(spramAddr[13:0], ?, False, ?);
				1: spram1.req(spramAddr[13:0], ?, False, ?);
			endcase
			bram_y.a.put(False, bramAddr1, ?);
			calcStage <= HIDDEN_MULT;
		end
		else if (iteration >= 50000 && iteration < 50400) begin //bias 1 calculation requests
			writeAddr <= unit_iteration;
			Bit#(9) bramAddr1 = access_iteration;
			bram_x.a.put(False, bramAddr1, ?);
			bram_y.a.put(False, bramAddr1, ?);
			spram1.req(spramAddr[13:0], ?, False, ?);
			calcStage <= BIAS1_ADD1;
		end
		else if (iteration >= 50400 && iteration < 50500) begin //bias 2 calculation requests
			writeAddr <= access_iteration;
			Bit#(9) bramAddr1 = access_iteration + 100;
			Bit#(9) bramAddr2 = access_iteration;
			bram_y.a.put(False, bramAddr1, ?);
			bram_carry.a.put(False, bramAddr2, ?);
			calcStage <= BIAS2_Mult;
		end
		else if (iteration >= 50500 && iteration < 50600) begin //bias 3 calculation requests
			writeAddr <= access_iteration;
			Bit#(9) bramAddr1 = access_iteration;
			Bit#(9) bramAddr2 = bramAddr1 + 200;
			bram_y.a.put(False, bramAddr1, ?);
			bram_y.b.put(False, bramAddr2, ?);
			calcStage <= BIAS3_Mult;
		end
		else if (iteration >= 50600 && iteration < 50700) begin //bias 4 calculation requests
			writeAddr <= access_iteration;
			
			Bit#(9) bramAddr1 = access_iteration;
			bram_carry.b.put(False, bramAddr1, ?);
			bram_hidden.b.put(False, bramAddr1, ?);
			calcStage <= BIAS4_ADD;
		end
		else if (iteration >= 50700 && iteration < 50800) begin //bias 5 calculation requests 
			writeAddr <= access_iteration;
			
			Bit#(9) bramAddr1 = access_iteration; 
			bram_carry.a.put(False, bramAddr1, ?);
			calcStage <= BIAS5_HS;
		end
		else if (iteration >= 50800 && iteration < 50900) begin //bias 6 calculation requests
			writeAddr <= access_iteration;
			
			Bit#(9) bramAddr1 = access_iteration;
			Bit#(9) bramAddr2 = access_iteration + 300;
			bram_hidden.a.put(False, bramAddr1, ?);
			bram_y.a.put(False, bramAddr2, ?);
			calcStage <= BIAS6_Mult;
		end
		else calcStage <= INCREMENT;
	endrule
	
	rule incrementer(calcStage == INCREMENT); //increment values
		`ifdef BSIM
			//$display("lstm1 increment");
		`endif
		if (iteration < 50400) begin 
			//alternate mask/increment spramAddr
			//input, 10000 iterations (100*25*4)
				// dequeue an input every 400 iterations (25 dequeues)
				// increment bramAddr1/ reset bramAddr1 every 400 iterations(25 resets)
				//+100 for i
				//+100 for f
				//+100 for c
				//+100 for o
				// increment writeAddr every 4 iterations/ reset writeAddr every 400 iterations (25 resets)
			//hidden, 40000 iterations (100*100*4)
				// increment bramAddr2 every 400 iterations (100 times)
				// increment bramAddr1/ reset bramAddr1 every 400 iterations(100 resets)
				//+100 for i 
				//+100 for f
				//+100 for c
				//+100 for o
				// increment writeAddr every 4 iterations/ reset writeAddr every 400 iterations (100 resets)
			//bias1, 100 iterations
			// increment bramAddr1/ reset bramAddr1 at the end
				//+100 for i
				//+100 for f
				//+100 for c
				//+100 for o
				// increment writeAddr every 4 iterations/ reset writeAddr at the end
				
			iteration <= iteration + 1;
			
			if (access_iteration < 399) access_iteration <= access_iteration + 1;
			else access_iteration <= 0; 
			
			if (unit_iteration < 99) begin
				if (layer_iteration < 3) layer_iteration <= layer_iteration + 1;
				else begin
					layer_iteration <= 0;
					unit_iteration <= unit_iteration + 1;
				end
			end
			else unit_iteration <= 0;
			
			case (mask) matches
				UPPER: begin
					mask <= LOWER;
				end
				LOWER: begin
					mask <= UPPER;
					spramAddr <= spramAddr + 1;
				end
			endcase
			calcStage <= FETCH;
		end
		else if (iteration >= 50400 && iteration < 50900) begin //bias cross-layer calculations
			// bias2 +100 for c_prev & f -> carry // start at 0 for c_prev and at 100 for f
			//bias3 +100 for i & c -> hidden // start at 0 for i and 200 for c
			//bias4 +100 for cf(carry) and ic(hidden) -> carry (c_state) //start at 0 for carry and hidden
			//bias5 +100 for carry (c_state) -> hidden (hs) // start at 0 for carry, same for hidden
			//bias6 +100 for hidden (hs) -> h_state & export // start at 0 for hidden
			iteration <= iteration + 1;
			if (access_iteration < 99) access_iteration <= access_iteration + 1;
			else access_iteration <= 0; 
			calcStage <= FETCH;
		end
		else begin
			access_iteration <= 0;
			iteration <= 0;
			if (height_count < 50) begin
				height_count <= height_count + 1;
				calcStage <= FETCH;
			end
			else calcStage <= FIN;
		end
	endrule
	
	method Action processInput(Bit#(8) in);
		`ifdef BSIM
			$display("processing LSTM1 input");
		`endif
		//input_counter <= input_counter + 1;
		inputQ.enq(unpack(in));
	endmethod
	
	method Action processWeight(Bit#(8) weight);
		`ifdef BSIM
			$display("processing LSTM1 weight");
		`endif
		Bit#(4) bytemask = 4'b1111;
		Bit#(16) maskeddata = 0;
		case (mask) matches
			UPPER: begin
				mask <= LOWER;
				bytemask = 4'b1100;
				maskeddata[15:8] = weight;
			end
			LOWER: begin
				mask <= UPPER;
				bytemask = 4'b0011;
				if (spramAddr < 25199) spramAddr <= spramAddr + 1;
				else spramAddr <= 0; 
				maskeddata[7:0] = weight;
			end
		endcase
		case (spramAddr[14]) matches
				0: spram0.req(spramAddr[13:0], maskeddata, True, bytemask);
				1: spram1.req(spramAddr[13:0], maskeddata, True, bytemask);
		endcase
	endmethod
	
	method Action processLSTM2Weight(Bit#(8) weight);
		lstm2.processWeight(weight);
	endmethod
	
	method Action processDenseWeight(Bit#(8) weight);
		lstm2.processDenseWeight(weight);
	endmethod
	
	method Bool inputReady;//ready to queue input
		return !inputQ.notEmpty;
	endmethod

	method Bool outputReady;//ready to dequeue output
		return outputQ.notEmpty;
	endmethod
	
	method Action start;
		if (calcStage == INIT) begin
			calcStage <= FETCH;
			`ifdef BSIM
				$display("lstm1 start");
			`endif
		end
	endmethod 

	method ActionValue#(Int#(8)) getOutput;
		outputQ.deq;
		return outputQ.first;
	endmethod


endmodule: mkLSTM1