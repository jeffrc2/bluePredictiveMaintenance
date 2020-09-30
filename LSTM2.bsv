import BRAMCore::*;
import FIFOF::*;
import BRAMFIFO::*;
import Dense::*;
import Spram::*;
import FIFO::*;
//invscale = 42;
//zeropoint = 0;

//lowerlimit = -1;//-2.5f
//upperlimit = 1;//2.5f
//onepoint = 102; //1.0f
//alpha = 20; //0.2f
//offset = 51;//0.5f

typedef enum {UPPER, LOWER} MaskHalf deriving (Bits,Eq);

typedef enum {INIT, 
	INPUT, 
	HIDDEN, 
	BIAS,	
	ACTIVATE1,	
	ACTIVATE2,	
	ACTIVATE3,	
	ACTIVATE4,	
	ACTIVATE5} Stage deriving (Bits,Eq);

interface LSTM2Ifc;
	method Action processInput(Bit#(8) in); //Put the input into the input queue.
	method Action processWeight(Bit#(8) weights); //Put the weights into the weights queue.
	method Action processDenseWeight(Bit#(8) weight);
	method Action start; 
	method Bool inputReady;
	method Bool outputReady;
	method ActionValue#(Int#(8)) getOutput;
endinterface

module mkLSTM2(LSTM2Ifc);

	DenseIfc dense <- mkDense; 

	BRAM_DUAL_PORT#(Bit#(9), Bit#(8)) bram_input <- mkBRAMCore2(100, False); //inputQ replaced with input BRAM in order to allow LSTM1  to continue running 
	
	FIFOF#(Int#(8)) outputQ <- mkSizedBRAMFIFOF(2);
	
	Int#(16) invscale = 42;
	Int#(16) zeropoint = 1;
	
	BRAM_DUAL_PORT#(Bit#(9), Bit#(8)) bram_hidden <- mkBRAMCore2(50, False);
	BRAM_DUAL_PORT#(Bit#(9), Bit#(8)) bram_carry <- mkBRAMCore2(50, False);
	
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
	//don't need secondary spram since it weights do not exceed capacity.
	
	BRAM_DUAL_PORT#(Bit#(9), Bit#(8)) bram_x <- mkBRAMCore2(200, False);
	BRAM_DUAL_PORT#(Bit#(9), Bit#(8)) bram_y <- mkBRAMCore2(200, False);
	
	Reg#(Bool) spramRead <- mkReg(False);
	Reg#(Bool) bramRead <- mkReg(False);

	function Int#(8) bramRespA(BRAM_DUAL_PORT#(Bit#(9), Bit#(8)) bram);
		return unpack(bram.a.read);
	endfunction
	
	function Int#(8) bramRespB(BRAM_DUAL_PORT#(Bit#(9), Bit#(8)) bram);
		return unpack(bram.b.read);
	endfunction
	
	rule relay(dense.outputReady);
		$display("lstm2 output relay");
		Int#(8) dataOut <- dense.getOutput;
		outputQ.enq(dataOut);
	endrule
	
	Reg#(Bit#(15)) mainIteration <- mkReg(0);//Range 0-30199 for input/hidden/bias, 30200-30449 for activate1-6
	Reg#(Bit#(14)) spramAddr <- mkReg(0);//Range 0-15099	
			
	Reg#(Bit#(9)) bramAddr <- mkReg(0);//Range 0-199 
	Reg#(Bit#(9)) hiddenAddr <- mkReg(0);//Range 0-49
	Reg#(Bit#(9)) unitCount <- mkReg(0);//(4*50 = 200 per unit) 0-99 for input, 100-149 for hidden, 150 for bias, (100 count units) 151-155 for activate1-5.
	Reg#(Bit#(6)) heightCount <- mkReg(0); //Range 0-49
	Reg#(Bit#(9)) inputCount <- mkReg(0); //Range 0-4999
	Reg#(Bit#(9)) inputAddr <- mkReg(0);
	
	Reg#(MaskHalf) incMask <- mkReg(UPPER); 
	Reg#(MaskHalf) calcMask <- mkReg(UPPER);
	
	function Int#(8) spramSelect(Bit#(16) spramValuePair);
		Int#(8) value = 0;
		case (calcMask) matches
			UPPER: value = unpack(spramValuePair[15:8]);
			LOWER: value = unpack(spramValuePair[7:0]);
		endcase
		return value;
	endfunction

	Reg#(Bit#(9)) writeAddr <- mkReg(0);	

	Reg#(Stage) fetchStage <- mkReg(INIT);
	FIFO#(Stage) fetchStageQ <- mkFIFO;
	Reg#(Stage) calcStage <- mkReg(INIT);

	Reg#(Bool) readX <- mkReg(False);
	Reg#(Bool) readY <- mkReg(False);
	Reg#(Bool) readHidden <- mkReg(False);

	Reg#(Bool) mainIncrementEN <- mkReg(False);
	Reg#(Bool) activateIncrementEN <- mkReg(False);	
	
	rule incrementMain(mainIncrementEN && mainIteration < 30200);
		`ifdef BSIM
			//$display("lstm2 increment main");
			//$display("lstm2 height ", heightCount);
			//$display("lstm2 iteration ", mainIteration);
		`endif
		//alternate mask/increment spramAddr
			//input, 20000 iterations (100*50*4)
				// increment bramAddr/ reset bramAddr every 200 iterations(100 resets)
				//+50 for i
				//+50 for f
				//+50 for c
				//+50 for o
				
		//increment iteration
		mainIteration <= mainIteration + 1;
		
		//transition stage
		if (mainIteration == 20000) begin
			fetchStage <= HIDDEN;
			fetchStageQ.enq(HIDDEN);
		end
		else if (mainIteration == 30000) begin
			fetchStage <= BIAS;
			fetchStageQ.enq(BIAS);
		end
		if (bramAddr < 199) bramAddr <= bramAddr + 1;
		else begin
			bramAddr <= 0;
			unitCount <= unitCount + 1;
		end
		
		case (incMask) matches
			UPPER: begin
				incMask <= LOWER;
			end
			LOWER: begin
				incMask <= UPPER;
				spramAddr <= spramAddr + 1;
			end
		endcase
	endrule
	
	rule incrementActivate(mainIncrementEN && mainIteration >= 30200 && mainIteration < 30449);
		`ifdef BSIM
			//$display("lstm2 increment activate");
			//$display("lstm2 height ", heightCount);
			//$display("lstm2 iteration ", mainIteration);
		`endif
		mainIteration <= mainIteration + 1;
		if (mainIteration < 30250) begin
			fetchStage <= ACTIVATE1;
			fetchStageQ.enq(ACTIVATE1);
		end
		else if (mainIteration < 30300) begin
			fetchStage <= ACTIVATE2;
			fetchStageQ.enq(ACTIVATE2);
		end
		else if (mainIteration < 30350) begin
			fetchStage <= ACTIVATE3;
			fetchStageQ.enq(ACTIVATE3);
		end
		else if (mainIteration < 30400) begin
			fetchStage <= ACTIVATE4;
			fetchStageQ.enq(ACTIVATE4);
		end
		else begin
			fetchStage <= ACTIVATE5;
			fetchStageQ.enq(ACTIVATE5);
			$display( "fetch going 5" );
		end
		if (bramAddr < 49) bramAddr <= bramAddr + 1;
		else begin
			bramAddr <= 0;
			unitCount <= unitCount + 1;
		end
	endrule
	
	rule resetMain(mainIncrementEN && mainIteration == 30449);//reset in the last iteration
		mainIteration <= 0;
		spramAddr <= 0;
		bramAddr <= 0;
		unitCount <= 0;
		readX <= False;
		readY <= False;
		hiddenAddr <= 0;
		inputCount <= 0;
		fetchStage <= INIT;
		fetchStageQ.enq(INIT);

		mainIncrementEN <= False;
		if (heightCount < 49) begin
			heightCount <= heightCount + 1;
		end
		else begin
			heightCount <= 0;
		end
	endrule
	
	rule cascadeFetchCalc;
		//calcStage <= fetchStage;
		fetchStageQ.deq;
		calcStage <= fetchStageQ.first;
		//$display( "going to %d", pack(fetchStage) );
		/*
		if ( fetchStage == ACTIVATE5 ) begin
			$display("going to 5");
		end
		if ( fetchStage == ACTIVATE1 ) begin
			$display("fetch activate 1");
		end
		*/
	endrule
	
	rule fetchInput(fetchStage == INPUT); //input calculation requests
		// dequeue an input every 400 iterations (25 dequeues)
		//`ifdef BSIM
			//$display("lstm2 input fetch ", inputCount);
		//`endif
		
		//fetch spram kernel value
		spram0.req(spramAddr[13:0], ?, False, ?);
		
		//fetch a new input at the beginning of each unit after the first, and enable reading BRAM X
		if (bramAddr == 0 && unitCount != 0) begin //dequeue input
			readX <= True;
		end
		if (bramAddr == 199) begin
			inputCount <= inputCount + 1;
		end
		bram_input.a.put(False, inputCount, ?);
		
		//fetch bram x value
		bram_x.a.put(False, bramAddr, ?);
		
		//update calc parameters
		calcMask <= incMask;
		writeAddr <= bramAddr;
	endrule
	
	rule calcInput(calcStage == INPUT); //100x4 steps before incrementing 
		//`ifdef BSIM
			//$display("lstm2 input");
		//`endif

		//receive bram x value
		Int#(8) aggregate = 0;
		if (readX) begin
			aggregate = bramRespA(bram_x);
		end
		//receive spram kernel value
		Bit#(16) valuepair <- spram0.resp;
		Int#(8) coeff = spramSelect(valuepair);
		//receive input value
		Int#(8) dataIn = bramRespA(bram_input);
		//calculate new x value
		Int#(8) product = quantizedMult(coeff, dataIn);
		bram_x.b.put(True, writeAddr, pack(quantizedAdd(aggregate, product)));
	endrule
	
	rule fetchHidden(fetchStage == HIDDEN); //input calculation requests
		// dequeue an input every 400 iterations (25 dequeues)
		//`ifdef BSIM
			//$display("lstm2 hidden fetch");
		//`endif
		
		// fetch spram recurrent value
		spram0.req(spramAddr[13:0], ?, False, ?);
		
		//fetch bram hidden and y value after the first hidden unit
		if (heightCount > 0) begin
			readHidden <= True; //Activates the hidden values after using zeroes initially
		end
		if (bramAddr == 199) begin //increment the hidden address
			hiddenAddr <= hiddenAddr + 1;
		end
		bram_hidden.a.put(False, hiddenAddr, ?);
		
		if (unitCount > 100) begin //Enables Y bram reads after the initial writes
			readY <= True;
		end
		
		//fetch bram y value
		bram_y.a.put(False, bramAddr, ?);
		
		//update calc parameters
		calcMask <= incMask;
		writeAddr <= bramAddr;
	endrule
	
	rule calcHidden(calcStage == HIDDEN);
		//`ifdef BSIM
			//$display("lstm2 hidden");
		//`endif
		
		//receive bram y value
		Int#(8) aggregate = 0;
		if (readY) begin
			aggregate = bramRespA(bram_y);
		end
		//receive spram recurrent value
		Bit#(16) valuepair = 0; 

		valuepair <- spram0.resp;

		Int#(8) coeff = spramSelect(valuepair);
		
		//receive stored hidden value
		Int#(8) hiddenIn = 0;
		if (readHidden) begin
			hiddenIn = bramRespA(bram_hidden);
		end
		
		//calculate new y value
		Int#(8) product = quantizedMult(coeff, hiddenIn);
		bram_x.b.put(True, writeAddr, pack(quantizedAdd(aggregate, product)));
	endrule
	
	
	rule fetchBias(fetchStage == BIAS); //input calculation requests
		//`ifdef BSIM
			//$display("lstm2 bias fetch" , bramAddr, mainIteration);
		//`endif
		//fetch bram x value
		bram_x.a.put(False, bramAddr, ?);
		//fetch bram y value
		bram_y.a.put(False, bramAddr, ?);
		//fetch spram value
		spram0.req(spramAddr[13:0], ?, False, ?);
		
		//update calc parameters
		calcMask <= incMask;
		writeAddr <= bramAddr;
	endrule
	
	Reg#(Int#(8)) bias <- mkReg(0);
	Reg#(Int#(8)) bias1 <- mkReg(0);

	Reg#(Bool) calcBias1 <- mkReg(False); //enables calcBiasZW
	Reg#(Bool) calcBias2 <- mkReg(False); //enables calcBiasHS
	
	Reg#(Bit#(9)) writeAddr1 <- mkReg(0);
	Reg#(Bit#(9)) writeAddr2 <- mkReg(0);
	
	rule calcBiasCascade;//follows fetchStage in a cascaded delay 
		calcBias1 <= (calcStage == BIAS);
		calcBias2 <= calcBias1;
	endrule

	rule calcBiasXY(calcStage == BIAS); 
		//`ifdef BSIM
			//$display("lstm2 bias ", writeAddr);
		//`endif
		//recieve spram bias value
		Bit#(16) valuepair <- spram0.resp;
		bias <= spramSelect(valuepair);
		
		//recieve bram X value
		Int#(8) x = bramRespA(bram_x);
		//recieve bram Y value
		Int#(8) y = bramRespA(bram_y);
		
		//calculate x + y
		bias1 <= quantizedAdd(x, y);
		writeAddr1 <= writeAddr;
	endrule
	
	Reg#(Int#(8)) bias2 <- mkReg(0);
	
	rule calcBiasZW(calcBias1); //
		// calculate x + y + bias
		bias2 <= quantizedAdd(bias1, bias);
		writeAddr2 <= writeAddr1;
	endrule
	
	rule calcBiasHS(calcBias2);
		// calculate hard_sigmoid(x + y+ bias)
		bram_y.b.put(True, writeAddr2, pack(hardSigmoid(bias2)));
	endrule
	
	
	rule fetchActivate1(fetchStage == ACTIVATE1); //bias 2 calculation requests
		//`ifdef BSIM
			//$display("lstm2 activate1 fetch", bramAddr);
		//`endif
		writeAddr <= bramAddr;
		Bit#(9) bramAddr1 = bramAddr*4+1; //F layer
		Bit#(9) bramAddr2 = bramAddr; //carry_prev
		bram_y.a.put(False, bramAddr1, ?);//fetch F from bram y
		bram_carry.a.put(False, bramAddr2, ?);//fetch carry_prev
	endrule
	
	rule calcActivate1(calcStage == ACTIVATE1);
		//`ifdef BSIM
			//$display("lstm2 activate1");
		//`endif
		Int#(8) c_prev = bramRespA(bram_carry); //retrieve previous carry
		Int#(8) f = bramRespA(bram_y); //retrieve f
		bram_carry.b.put(True, writeAddr, pack(quantizedMult(c_prev,f))); //carry[j] = hard_sigmoid_func(f[j])*carry[j]
	endrule
	
	rule fetchActivate2(fetchStage == ACTIVATE2); //bias 3 calculation requests
		//`ifdef BSIM
			//$display("lstm2 activate2 fetch", bramAddr);
		//`endif
		writeAddr <= bramAddr;
		Bit#(9) bramAddr1 = bramAddr*4; // I layer
		Bit#(9) bramAddr2 = bramAddr1 + 2; // C layer
		bram_y.a.put(False, bramAddr1, ?); 
		bram_y.b.put(False, bramAddr2, ?);
	endrule
	
	rule calcActivate2(calcStage == ACTIVATE2);
		//`ifdef BSIM
			//$display("lstm2 activate2");
		//`endif
		Int#(8) i = bramRespA(bram_y); 
		Int#(8) c = bramRespB(bram_y);
		bram_hidden.b.put(True, writeAddr, pack(quantizedMult(i,c))); //hidden[j] = hard_sigmoid_func(i[j])*hard_sigmoid_func(c[j])
	endrule
	
	rule fetchActivate3(fetchStage == ACTIVATE3); //bias 4 calculation requests
		//`ifdef BSIM
			//$display("lstm2 activate3 fetch", bramAddr);
		//`endif
		writeAddr <= bramAddr;
		bram_carry.a.put(False, bramAddr, ?); // Fetch F*c_prev
		bram_hidden.a.put(False, bramAddr, ?);// Fetch I*C
	endrule
	
	rule calcActivate3(calcStage == ACTIVATE3);
		//`ifdef BSIM
			//$display("lstm2 activate3");
		//`endif
		Int#(8) cf = bramRespA(bram_carry);
		Int#(8) ic = bramRespA(bram_hidden);
		bram_carry.b.put(True, writeAddr, pack(quantizedAdd(cf,ic))); //new carry state: carry[j] = carry[j] + hidden[j]
	endrule
	
	rule fetchActivate4(fetchStage == ACTIVATE4); //bias 5 calculation requests 
		//`ifdef BSIM
			//$display("lstm2 activate4 fetch", bramAddr);
		//`endif
		writeAddr <= bramAddr;
		bram_carry.a.put(False, bramAddr, ?); //Fetch I*F*C*c_prev
	endrule
	
	rule calcActivate4(calcStage == ACTIVATE4);
		//`ifdef BSIM
			//$display("lstm2 activate4", bramAddr);
		//`endif
		Int#(8) c_state = bramRespA(bram_carry);
		bram_hidden.b.put(True, writeAddr, pack(hardSigmoid(c_state))); //hidden[j] = hard_sigmoid(carry[j])
	endrule
	
	rule fetchActivate5(fetchStage == ACTIVATE5); //bias 6 calculation requests
		`ifdef BSIM
			$display("lstm2 activate5 fetch ", bramAddr, " ", heightCount);
		`endif
		writeAddr <= bramAddr; 
		Bit#(9) bramAddr1 = bramAddr; // hard_sigmoid(I*F*C*c_prev)
		Bit#(9) bramAddr2 = bramAddr*4 + 3; // O layer
		bram_hidden.a.put(False, bramAddr1, ?); //fetch hard_sigmoid
		bram_y.a.put(False, bramAddr2, ?); //fetch O layer
	endrule	
	
	Reg#(Bool) outputActivate <- mkReg(False);
	
	rule outputCascade(heightCount == 49 && fetchStage == ACTIVATE5);//enables export in the final timestep
		outputActivate <= True;
		$display("output activate ");
	endrule

	rule calcActivate5(calcStage == ACTIVATE5); //must wait for LSTM2 queue to be empty before allowed to continue.
		//`ifdef BSIM
			//$display("lstm2 activate5", bramAddr, " ", outputActivate);
		//`endif
		
		Int#(8) hs = bramRespA(bram_hidden);
		Int#(8) o = bramRespA(bram_y);
		let h_state = quantizedMult(hs, o);
		bram_hidden.b.put(True, writeAddr, pack(h_state)); //new hidden state: hidden[j] = hidden[j]*o[j];
		if (outputActivate) begin
			`ifdef BSIM
				$display("dense reached");
				//$finish(0);
			`endif
			dense.processInput(h_state);
			dense.start;
		end
	endrule 
	
	method Action processInput(Bit#(8) in);
		`ifdef BSIM
			$display("processing LSTM2 input");
		`endif
		if (inputAddr < 99) inputAddr <= inputAddr + 1;
		else begin
			inputAddr <= 0;
		end
		bram_input.b.put(True, inputAddr, in);
	endmethod
	
	method Action processWeight(Bit#(8) weight);
		`ifdef BSIM
			$display("processing LSTM2 weight");
		`endif
		Bit#(4) bytemask = 0;
		Bit#(16) maskeddata = 0;
		case (incMask) matches
			UPPER: begin
				incMask <= LOWER;
				bytemask = 4'b1100;
				maskeddata[15:8] = weight;
			end
			LOWER: begin
				incMask <= UPPER;
				bytemask = 4'b0011;
				
				if (spramAddr < 15099) spramAddr <= spramAddr + 1;
				else spramAddr <= 0; 
				maskeddata[7:0] = weight;
			end
		endcase
		spram0.req(spramAddr, maskeddata, True, bytemask);
	endmethod
	
	method Action processDenseWeight(Bit#(8) weight);
		dense.processWeight(weight);
	endmethod
	
	method Bool inputReady;//ready to queue input
		return (inputAddr != 99);
	endmethod
	
	method Bool outputReady;//ready to dequeue output
		return outputQ.notEmpty;
	endmethod
	
	method Action start;
		mainIncrementEN <= True;
		if (calcStage == INIT) begin
			fetchStage <= INPUT;
			fetchStageQ.enq(INPUT);
			`ifdef BSIM
				$display("lstm2 start");
			`endif
		end
	endmethod 
	
	method ActionValue#(Int#(8)) getOutput;
		outputQ.deq;
		return outputQ.first;
	endmethod

endmodule: mkLSTM2
