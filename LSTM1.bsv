import BRAMCore::*;
import FIFOF::*;
import BRAMFIFO::*;
//import LSTM2::*;
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
//spram-required
	INPUT, 
	HIDDEN, 
	BIAS,
	ACTIVATE1,
	ACTIVATE2,
	ACTIVATE3,
	ACTIVATE4,
	ACTIVATE5} Stage deriving (Bits,Eq);

interface LSTM1Ifc;
	method Action processInput(Bit#(8) in); //Put the input into the input queue.
	method Action processWeight(Bit#(8) weight); //Put the weights into the weights queue.
	method Action processLSTM2Weight(Bit#(8) weight);
	method Action processDenseWeight(Bit#(8) weight); 
	method Bool inputReady;
	method Bool outputReady;
	method Bool lstmRunning;
	method ActionValue#(Int#(8)) getOutput;
endinterface

module mkLSTM1(LSTM1Ifc);

	//LSTM2Ifc lstm2 <- mkLSTM2; 

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
	
	
	BRAM_DUAL_PORT#(Bit#(9), Bit#(8)) bram_x <- mkBRAMCore2(400, False);
	BRAM_DUAL_PORT#(Bit#(9), Bit#(8)) bram_y <- mkBRAMCore2(400, False);
	
	Reg#(Bool) spramRead <- mkReg(False);
	Reg#(Bool) bramRead <- mkReg(False);
	
	function Int#(8) bramRespA(BRAM_DUAL_PORT#(Bit#(9), Bit#(8)) bram);
		return unpack(bram.a.read);
	endfunction
	
	function Int#(8) bramRespB(BRAM_DUAL_PORT#(Bit#(9), Bit#(8)) bram);
		return unpack(bram.b.read);
	endfunction
	
	//rule relay(lstm2.outputReady);
		//$display("lstm1 output relay");
		//Int#(8) dataOut <- lstm2.getOutput;
		//outputQ.enq(dataOut);
	//endrule
	
	Reg#(Bit#(16)) mainIteration <- mkReg(0);//Range 0-50399 for input/hidden/bias, 50400-50899 for activate1-6
	Reg#(Bit#(15)) spramAddr <- mkReg(0);//Range 0-25199
	
	Reg#(Bit#(9)) bramAddr <- mkReg(0);//Range 0-399
	
	Reg#(Bit#(9)) hiddenAddr <- mkReg(0);//Range 0-99
	Reg#(Bit#(9)) unitCount <- mkReg(0); //(4*100 = 400 per unit) 0-24 for input, 25-124 for hidden, 125 for bias, (100 count units) 126-130 for activate1-5.
	Reg#(Bit#(6)) heightCount <- mkReg(0); //Range 0-49
	Reg#(Bit#(11)) inputCount <- mkReg(0); //Range 0-1249
	
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
	
	FIFOF#(Bit#(9)) writeAddrQ <- mkSizedBRAMFIFOF(3);
	//Reg#(Bit#(9)) writeAddr <- mkReg(0);

	Reg#(Stage) fetchStage <- mkReg(INPUT);
	FIFOF#(Stage) fetchStageQ <- mkSizedBRAMFIFOF(3);
	FIFOF#(Stage) calcStageQ <- mkSizedBRAMFIFOF(3);
	Reg#(Stage) calcStage <- mkReg(INIT);
	
	Reg#(Bool) readX <- mkReg(False);
	Reg#(Bool) readY <- mkReg(False);
	Reg#(Bool) readHidden <- mkReg(False);
	
	Reg#(Bool) mainIncrementEN <- mkReg(False);
	Reg#(Bool) activateIncrementEN <- mkReg(False);
		
	function Action setStage(Stage newStage);
		action
			fetchStage <= newStage;
			calcStageQ.enq(newStage);
		endaction
	endfunction
		
	rule incrementMain(mainIncrementEN && mainIteration < 50400);
		`ifdef BSIM
			$display("lstm1 increment main ", mainIteration, " height ", heightCount);
		`endif
		//alternate mask/increment spramAddr
			//input, 10000 iterations (25*100*4)
				
				// increment bramAddr1/ reset bramAddr1 every 400 iterations(25 resets)
				//+100 for i
				//+100 for f
				//+100 for c
				//+100 for o
				
		//increment iteration
		mainIteration <= mainIteration + 1;
		
		
		//transition stage
		if (mainIteration == 10000) begin
			setStage(HIDDEN);
		end
		else if (mainIteration == 50000) begin
			setStage(BIAS);
		end
		
		if (bramAddr < 399) bramAddr <= bramAddr + 1;
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
	
	rule incrementActivate(mainIncrementEN && mainIteration >= 50400 && mainIteration < 50899);
		`ifdef BSIM
			$display("lstm1 increment activate ", mainIteration, " height ", heightCount);
		`endif
		mainIteration <= mainIteration + 1;
		if (mainIteration == 50400) begin
			setStage(ACTIVATE1);
		end
		else if (mainIteration == 50500) begin
			setStage(ACTIVATE2);
		end
		else if (mainIteration == 50600) begin
			setStage(ACTIVATE3);
		end
		else if (mainIteration == 50700) begin
			setStage(ACTIVATE4);
		end
		else if (mainIteration == 50800) begin
			setStage(ACTIVATE5);
		end
		if (bramAddr < 99) bramAddr <= bramAddr + 1;
		else begin
			bramAddr <= 0;
			unitCount <= unitCount + 1;
		end
	endrule
	
	rule resetMain(mainIncrementEN && mainIteration == 50899);//reset in the last iteration
		`ifdef BSIM
			$display("lstm1 increment reset ", mainIteration, " height ", heightCount);
		`endif
		mainIteration <= 0;
		spramAddr <= 0;
		bramAddr <= 0;
		unitCount <= 0;
		readX <= False;
		readY <= False;
		hiddenAddr <= 0;
		if (heightCount < 49) begin
			heightCount <= heightCount + 1;
			fetchStage <= INPUT;
		end
		else begin
			mainIncrementEN <= False;
			fetchStage <= INIT;
			heightCount <= 0;

		end
	endrule
	
	rule cascadeCalc(calcStageQ.notEmpty);
		calcStage <= calcStageQ.first;
		calcStageQ.deq;
		//calcStage <= fetchStage;
	endrule
	
	Reg#(Int#(8)) inputReg <- mkReg(0);
	
	rule fetchInput(fetchStage == INPUT); //input calculation requests
		// dequeue an input every 200 iterations (25 dequeues)
		`ifdef BSIM
			$display("lstm1 input fetch spram ", spramAddr, "bram ", bramAddr);
		`endif
		
		//fetch spram kernel value
		spram0.req(spramAddr[13:0], ?, False, ?);
		
		inputReg <= inputQ.first;
		//fetch a new input at the beginning of each unit after the first, and enable reading BRAM X
		if (bramAddr == 399) begin //dequeue input
			if (inputQ.notEmpty) begin
				inputQ.deq;
			end else begin
				$display("inputQ is currently empty ", inputCount);
				$finish(0);
			end
		end
		if (unitCount == 1 && bramAddr == 0) readX <= True;
		
		//fetch bram x value
		bram_x.a.put(False, bramAddr, ?);
		
		//update calc parameters
		calcMask <= incMask;
		writeAddrQ.enq(bramAddr);
		//writeAddr <= bramAddr;
	endrule
	
	rule calcInput(calcStage == INPUT); //100x4 steps before incrementing 
		`ifdef BSIM
			$display("lstm1 input write ", writeAddrQ.first);
		`endif

		//receive bram x value
		Int#(8) aggregate = 0;
		if (readX) begin
			aggregate = bramRespA(bram_x);
		end
		//receive spram kernel value
		Bit#(16) valuepair <- spram0.resp;
		Int#(8) coeff = spramSelect(valuepair);
		//receive input value
		Int#(8) dataIn = inputReg;
		//calculate new x value
		Int#(8) product = quantizedMult(coeff, dataIn);
		
		Bit#(9) writeAddr = writeAddrQ.first;
		writeAddrQ.deq;
		bram_x.b.put(True, writeAddr, pack(quantizedAdd(aggregate, product)));
	endrule
	
	Reg#(Bit#(1)) spramTop <- mkReg(0);
	
	rule fetchHidden(fetchStage == HIDDEN); //input calculation requests
		// dequeue an input every 400 iterations (25 dequeues)
		`ifdef BSIM
			$display("lstm1 hidden fetch");
		`endif
		
		// fetch spram recurrent value
		case (spramAddr[14]) matches
			0: spram0.req(spramAddr[13:0], ?, False, ?);
			1: spram1.req(spramAddr[13:0], ?, False, ?);
		endcase
		
		spramTop <= spramAddr[14];
		
		//fetch bram hidden and y value after the first hidden unit
		if (heightCount > 0) begin
			readHidden <= True; //Activates the hidden values after using zeroes initially
		end
		if (bramAddr == 399) begin //increment the hidden address
			hiddenAddr <= hiddenAddr + 1;
		end
		bram_hidden.a.put(False, hiddenAddr, ?);
		
		if (unitCount > 25) begin //Enables Y bram reads after the initial writes
			readY <= True;
		end
		
		//fetch bram y value
		bram_y.a.put(False, bramAddr, ?);
		
		//update calc parameters
		calcMask <= incMask;
		writeAddrQ.enq(bramAddr);
		//writeAddr <= bramAddr;
	endrule
	
	rule calcHidden(calcStage == HIDDEN);
		`ifdef BSIM
			$display("lstm1 hidden");
		`endif
		
		//receive bram y value
		Int#(8) aggregate = 0;
		if (readY) begin
			aggregate = bramRespA(bram_y);
		end
		//receive spram recurrent value
		Bit#(16) valuepair = 0; 
		case (spramTop) matches 
			0: valuepair <- spram0.resp;
			1: valuepair <- spram1.resp;
		endcase
		Int#(8) coeff = spramSelect(valuepair);
		
		//receive stored hidden value
		Int#(8) hiddenIn = 0;
		if (readHidden) begin
			hiddenIn = bramRespA(bram_hidden);
		end
		
		//calculate new y value
		Int#(8) product = quantizedMult(coeff, hiddenIn);
		
		Bit#(9) writeAddr = writeAddrQ.first;
		writeAddrQ.deq;
		bram_x.b.put(True, writeAddr, pack(quantizedAdd(aggregate, product)));
	endrule

	
	rule fetchBias(fetchStage == BIAS); //input calculation requests
		`ifdef BSIM
			$display("lstm1 bias fetch");
		`endif
		//fetch bram x value
		bram_x.a.put(False, bramAddr, ?);
		//fetch bram y value
		bram_y.a.put(False, bramAddr, ?);
		//fetch spram value
		spram1.req(spramAddr[13:0], ?, False, ?);
		
		//update calc parameters
		calcMask <= incMask;
		writeAddrQ.enq(bramAddr);
		//writeAddr <= bramAddr;
	endrule

	Reg#(Int#(8)) bias <- mkReg(0);
	Reg#(Int#(8)) bias1 <- mkReg(0);
	
	Reg#(Bool) calcBias1 <- mkReg(False); //enables calcBiasZW
	Reg#(Bool) calcBias2 <- mkReg(False); //enables calcBiasHS
	
	FIFOF#(Bit#(9)) writeAddrBiasQ <- mkSizedBRAMFIFOF(3);
	//Reg#(Bit#(9)) writeAddr1 <- mkReg(0);
	//Reg#(Bit#(9)) writeAddr2 <- mkReg(0);
	
	rule calcBiasCascade;//follows fetchStage in a cascaded delay 
		calcBias1 <= (calcStage == BIAS);
		calcBias2 <= calcBias1;
	endrule

	rule calcBiasXY(calcStage == BIAS); 
		`ifdef BSIM
			$display("lstm1 bias");
		`endif
		//recieve spram bias value
		Bit#(16) valuepair <- spram1.resp;
		bias <= spramSelect(valuepair);
		
		//recieve bram X value
		Int#(8) x = bramRespA(bram_x);
		//recieve bram Y value
		Int#(8) y = bramRespA(bram_y);
		
		//calculate x + y
		bias1 <= quantizedAdd(x, y);
		Bit#(9) writeAddr = writeAddrQ.first;
		writeAddrQ.deq;
		writeAddrBiasQ.enq(writeAddr);
		//writeAddr1 <= writeAddr;
	endrule

	Reg#(Int#(8)) bias2 <- mkReg(0);
	
	rule calcBiasZW(calcBias1); //
		// calculate x + y + bias
		bias2 <= quantizedAdd(bias1, bias);

		//writeAddr2 <= writeAddr1;
	endrule
	
	rule calcBiasHS(calcBias2);
		// calculate hard_sigmoid(x + y+ bias)
		writeAddrBiasQ.deq;
		Bit#(9) writeAddr = writeAddrBiasQ.first;
		bram_y.b.put(True, writeAddr, pack(hardSigmoid(bias2)));
		//bram_y.b.put(True, writeAddr2, pack(hardSigmoid(bias2)));
	endrule

	
	rule fetchActivate1(fetchStage == ACTIVATE1); //bias 2 calculation requests
		`ifdef BSIM
			$display("lstm1 activate1 fetch");
		`endif
		writeAddrQ.enq(bramAddr);//writeAddr <= bramAddr;
		Bit#(9) bramAddr1 = bramAddr*4+1; //F layer
		Bit#(9) bramAddr2 = bramAddr; //carry_prev
		bram_y.a.put(False, bramAddr1, ?);//fetch F from bram y
		bram_carry.a.put(False, bramAddr2, ?);//fetch carry_prev
	endrule
	
	rule calcActivate1(calcStage == ACTIVATE1);
		`ifdef BSIM
			$display("lstm1 activate1");
		`endif
		Int#(8) c_prev = bramRespA(bram_carry); //retrieve previous carry
		Int#(8) f = bramRespA(bram_y); //retrieve f
		Bit#(9) writeAddr = writeAddrQ.first;
		writeAddrQ.deq;
		bram_carry.b.put(True, writeAddr, pack(quantizedMult(c_prev,f))); //carry[j] = hard_sigmoid_func(f[j])*carry[j]
	endrule
	
	rule fetchActivate2(fetchStage == ACTIVATE2); //bias 3 calculation requests
		`ifdef BSIM
			$display("lstm1 activate2 fetch");
		`endif
		writeAddrQ.enq(bramAddr);//writeAddr <= bramAddr;
		Bit#(9) bramAddr1 = bramAddr*4; // I layer
		Bit#(9) bramAddr2 = bramAddr1 + 2; // C layer
		bram_y.a.put(False, bramAddr1, ?); 
		bram_y.b.put(False, bramAddr2, ?);
	endrule
	
	rule calcActivate2(calcStage == ACTIVATE2);
		`ifdef BSIM
			$display("lstm1 activate2");
		`endif
		Int#(8) i = bramRespA(bram_y); 
		Int#(8) c = bramRespB(bram_y);
		Bit#(9) writeAddr = writeAddrQ.first;
		writeAddrQ.deq;
		bram_hidden.b.put(True, writeAddr, pack(quantizedMult(i,c))); //hidden[j] = hard_sigmoid_func(i[j])*hard_sigmoid_func(c[j])
	endrule
	
	rule fetchActivate3(fetchStage == ACTIVATE3); //bias 4 calculation requests
		`ifdef BSIM
			$display("lstm1 activate3 fetch");
		`endif
		writeAddrQ.enq(bramAddr);//writeAddr <= bramAddr;
		bram_carry.a.put(False, bramAddr, ?); // Fetch F*c_prev
		bram_hidden.a.put(False, bramAddr, ?);// Fetch I*C
	endrule
	
	rule calcActivate3(calcStage == ACTIVATE3);
		`ifdef BSIM
			$display("lstm1 activate3");
		`endif
		Int#(8) cf = bramRespA(bram_carry);
		Int#(8) ic = bramRespA(bram_hidden);
		Bit#(9) writeAddr = writeAddrQ.first;
		writeAddrQ.deq;
		bram_carry.b.put(True, writeAddr, pack(quantizedAdd(cf,ic))); //new carry state: carry[j] = carry[j] + hidden[j]
	endrule
	
	rule fetchActivate4(fetchStage == ACTIVATE4); //bias 5 calculation requests 
		`ifdef BSIM
			$display("lstm1 activate4 fetch");
		`endif
		writeAddrQ.enq(bramAddr);//writeAddr <= bramAddr;
		bram_carry.a.put(False, bramAddr, ?); //Fetch I*F*C*c_prev
	endrule
	
	rule calcActivate4(calcStage == ACTIVATE4);
		`ifdef BSIM
			$display("lstm1 activate4");
		`endif
		Int#(8) c_state = bramRespA(bram_carry);
		Bit#(9) writeAddr = writeAddrQ.first;
		writeAddrQ.deq;
		bram_hidden.b.put(True, writeAddr, pack(hardSigmoid(c_state))); //hidden[j] = hard_sigmoid(carry[j])
	endrule
	
	
	rule fetchActivate5(fetchStage == ACTIVATE5); //bias 6 calculation requests
		`ifdef BSIM
			$display("lstm1 activate5 fetch");
		`endif
		writeAddrQ.enq(bramAddr);//writeAddr <= bramAddr; 
		Bit#(9) bramAddr1 = bramAddr; // hard_sigmoid(I*F*C*c_prev)
		Bit#(9) bramAddr2 = bramAddr*4 + 3; // O layer
		bram_hidden.a.put(False, bramAddr1, ?); //fetch hard_sigmoid
		bram_y.a.put(False, bramAddr2, ?); //fetch O layer
	endrule
	
	rule calcActivate5(calcStage == ACTIVATE5);
		`ifdef BSIM
			$display("lstm1 activate5");
		`endif
		Int#(8) hs = bramRespA(bram_hidden);
		Int#(8) o = bramRespA(bram_y);
		let h_state = quantizedMult(hs, o);
		Bit#(9) writeAddr = writeAddrQ.first;
		writeAddrQ.deq;
		bram_hidden.b.put(True, writeAddr, pack(h_state)); //new hidden state: hidden[j] = hidden[j]*o[j];
		outputQ.enq(h_state);
		//lstm2.processInput(pack(h_state));
		//lstm2.start;
	endrule 
	
	method Action processInput(Bit#(8) in);
		`ifdef BSIM
			$display("processing LSTM1 input", inputCount);
		`endif
		if (mainIncrementEN == False) begin
			mainIncrementEN <= True;
		end
		inputCount <= inputCount + 1;
		inputQ.enq(unpack(in));
	endmethod
	
	method Action processWeight(Bit#(8) weight);
		`ifdef BSIM
			$display("processing LSTM1 weight");
		`endif
		Bit#(4) bytemask = 4'b1111;
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
		//lstm2.processWeight(weight);
	endmethod
	
	method Action processDenseWeight(Bit#(8) weight);
		//lstm2.processDenseWeight(weight);
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


endmodule: mkLSTM1