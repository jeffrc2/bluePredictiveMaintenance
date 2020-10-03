//Default packages
import FIFOF::*;
import BRAMCore::*;
import BRAMFIFO::*;
//up5k packages
import Spram::*;
import DSPArith::*;
//project packages
import LSTM2::*;

typedef enum {UPPER, LOWER} MaskHalf deriving (Bits,Eq);

typedef enum {INIT = 0, 
	INPUT, 
	HIDDEN, 
	BIAS,
	ACTIVATE0,
	ACTIVATE1,
	ACTIVATE2,
	ACTIVATE3,
	ACTIVATE4} Stage deriving (Bits,Eq);//INPUT, HIDDEN, BIAS require weights from the SPRAM, while the ACTIVATEs do not.
	
typedef enum {INIT, MAIN, ACTIVATE} State deriving (Bits,Eq);

interface LSTM1Ifc;
	method Action processWeight(Bit#(8) weight); //Put the weights into the weights queue.
	method Action processLSTM2Weight(Bit#(8) weight);
	method Action processDenseWeight(Bit#(8) weight);
	method Action processInput(Bit#(8) in); //Put the input into the input queue.
 
	method Bool inputReady;
	method Bool outputReady;
	method ActionValue#(Int#(8)) getOutput;
endinterface

module mkLSTM1(LSTM1Ifc);

	LSTM2Ifc lstm2 <- mkLSTM2;

	Reg#(Bool) run <- mkReg(False);

	FIFOF#(Int#(8)) inputQ <- mkSizedBRAMFIFOF(5);
	FIFOF#(Int#(8)) outputQ <- mkSizedBRAMFIFOF(1);
	
	//weight storage
	Spram256KAIfc spram0 <- mkSpram256KA;
	Spram256KAIfc spram1 <- mkSpram256KA;
	
	//hidden and carry states
	BRAM_DUAL_PORT#(Bit#(9), Bit#(8)) bramH <- mkBRAMCore2(100, False);
	BRAM_DUAL_PORT#(Bit#(9), Bit#(8)) bramC <- mkBRAMCore2(100, False);
	//work memory
	BRAM_DUAL_PORT#(Bit#(9), Bit#(8)) bramX <- mkBRAMCore2(400, False);
	BRAM_DUAL_PORT#(Bit#(9), Bit#(8)) bramY <- mkBRAMCore2(400, False);
	
    Reg#(State) state <- mkReg(INIT);
	
	rule start(inputQ.notEmpty);
		run <= True;
	endrule

//increment 
	Reg#(MaskHalf) readMask <- mkReg(UPPER);//Exclusively used by increment_main
	Reg#(Bit#(15)) spramAddrR <- mkReg(0);//Range 0-25199
	Reg#(Bit#(9)) bramAddrR <- mkReg(0);//Range 0-399
	
	FIFOF#(Stage) fetchQ <- mkSizedBRAMFIFOF(2);
	FIFOF#(Stage) calcQ <- mkSizedBRAMFIFOF(2);
	
	function Action setStage(Stage newStage);
		action
			fetchQ.enq(newStage);
			calcQ.enq(newStage);
		endaction
	endfunction
	
	FIFOF#(Bit#(9)) bramAddrFetchQ <- mkSizedBRAMFIFOF(2);
	FIFOF#(MaskHalf) maskCalcQ <- mkSizedBRAMFIFOF(4);
	FIFOF#(Bit#(15)) spramAddrFetchQ <- mkSizedBRAMFIFOF(2);
	
	FIFOF#(Bit#(9)) bramAddrCalcQ <- mkSizedBRAMFIFOF(2);
	
	rule increment_main(run && state == MAIN);
		//enqueue fetch and calc stages
		Stage stg = HIDDEN;
		if (spramAddrR < 5000) stg = INPUT;
		else if (spramAddrR >= 25000) stg = BIAS;
		setStage(stg);
		
		//enqueue fetch, spramAddr, bramAddr, mask values
		spramAddrFetchQ.enq(spramAddrR);
		bramAddrFetchQ.enq(bramAddrR);
		maskCalcQ.enq(readMask);
		
		//update the bramAddr
		Bit#(9) bramAddrNew = bramAddrR + 1;
		if (bramAddrR == 399) bramAddrNew = 0;
		
		//update the bytemask and spramAddr
		MaskHalf maskUpdate = UPPER;
		Bit#(15) spramAddrNew = spramAddrR;
		case (readMask) matches 
			UPPER: begin
				maskUpdate = LOWER;
			end
			LOWER: begin
				maskUpdate = UPPER;
				if (spramAddrR < 25199) begin
                    spramAddrNew = spramAddrR + 1;
				end
				else begin //very last weight, send to activation stage
					spramAddrNew = 0;
					state <= ACTIVATE;
				end
			end
		endcase
		//update spramAddr, bramAddr, maskCalcQ registers
		spramAddrR <= spramAddrNew;
		readMask <= maskUpdate;
		bramAddrR <= bramAddrNew;
	endrule
	
	Reg#(Bit#(3)) activateCount <- mkReg(0);
	
	rule increment_activate(run && state == ACTIVATE && activateCount < 5);
	
		bramAddrFetchQ.enq(bramAddrR);
	
		Bit#(9) bramAddrNew = bramAddrR + 1;
		if (bramAddrR == 99) begin
			bramAddrNew = 0;
			activateCount <= activateCount + 1;
		end
		bramAddrR <= bramAddrNew;
		
		Stage stg = ACTIVATE0;
		if (activateCount == 1) stg = ACTIVATE1;
		else if (activateCount == 2) stg = ACTIVATE2;
		else if (activateCount == 3) stg = ACTIVATE3;
		else if (activateCount == 4) stg = ACTIVATE4;
		setStage(stg);
		
	endrule
	
	Reg#(Bit#(6)) heightCount <- mkReg(0);
	
	rule reset(run && state == ACTIVATE && activateCount == 5);
		State st = MAIN;
		Bit#(6) height = heightCount + 1;
		if (height == 50) begin
			height = 0;
			st = INIT;
			run <= False;//turn off processing since it is done
		end
		state <= st;
		heightCount <= height;
		activateCount <= 0;
	endrule
	
	function Action bramReqA(Bit#(9) bram_addr, BRAM_DUAL_PORT#(Bit#(9), Bit#(8)) bram_port);
		action
			bram_port.a.put(False, bram_addr, ?);
		endaction
	endfunction
	
	function Action bramReqB(Bit#(9) bram_addr, BRAM_DUAL_PORT#(Bit#(9), Bit#(8)) bram_port);
		action
			bram_port.b.put(False, bram_addr, ?);
		endaction
	endfunction
	
	function Action spramReq(Bit#(14) spram_addr, Spram256KAIfc spram);
		action
			spram.req(spram_addr, ?, False, ?);
		endaction
	endfunction
	
	FIFOF#(Bool) readQ <- mkSizedBRAMFIFOF(2);

	rule fetchInput(fetchQ.first == INPUT);
		fetchQ.deq;
		//request SPRAM weight
		Bit#(15) spramAddr = spramAddrFetchQ.first; 
		spramReq(spramAddr[13:0], spram0);
		spramAddrFetchQ.deq;
		
		//indicate read status for bramX
		Bool readX = True;
		if (spramAddr < 100) readX = False; //enable readX after first batch, but is reset at each timestep
		readQ.enq(readX);
		
		//request BRAM value
		Bit#(9) bramAddr = bramAddrFetchQ.first;
		bramReqA(bramAddr, bramX); //bramX.a.put(False, bramAddr, ?);
		bramAddrFetchQ.deq;
		bramAddrCalcQ.enq(bramAddr);
	endrule
    
	FIFOF#(Bit#(1)) spramTopQ <- mkSizedBRAMFIFOF(4);
	Reg#(Bit#(9)) hiddenAddr <- mkReg(0);
	Reg#(Bool) readHidden <- mkReg(False);
	FIFOF#(Bool) readHiddenQ <- mkSizedBRAMFIFOF(2);
	
	rule fetchHidden(fetchQ.first == HIDDEN);
		fetchQ.deq;
		//request SPRAM weight
		Bit#(15) spramAddr = spramAddrFetchQ.first; 
		case (spramAddr[14]) matches
			0: spramReq(spramAddr[13:0], spram0);
			1: spramReq(spramAddr[13:0], spram1);
		endcase
		spramAddrFetchQ.deq;
		
		//pass along spram Unit
		spramTopQ.enq(spramAddr[14]); 
		
		//indicate read status for bramX
		Bool readY = True;
		Bool hidden = readHidden;
		if (spramAddr < 5200) readY = False; //enable readY after first batch, but is reset at each timestep
		else if (!readHidden) begin //enable hidden read on for all units after the first, no matter the timestep
			hidden = True;
			readHidden <= hidden;
		end
		readQ.enq(readY);
		readHiddenQ.enq(hidden);
		
		//request BRAM value
		Bit#(9) bramAddr = bramAddrFetchQ.first;
		bramReqA(bramAddr, bramY); //bramX.a.put(False, bramAddr, ?);
		bramReqA(hiddenAddr, bramH);
		if (bramAddr == 399) begin
			if (hiddenAddr < 99) hiddenAddr <= hiddenAddr + 1;
			else hiddenAddr <= 0;
		end
		bramAddrFetchQ.deq;
		bramAddrCalcQ.enq(bramAddr);
	endrule
	
	rule fetchBias(fetchQ.first == BIAS);
		fetchQ.deq;
		//request SPRAM weight
		Bit#(15) spramAddr = spramAddrFetchQ.first; 
		spramReq(spramAddrR[13:0], spram1);
		spramAddrFetchQ.deq;
	
		//request BRAM values
		Bit#(9) bramAddr = bramAddrFetchQ.first;
		bramReqA(bramAddr, bramX);
		bramReqA(bramAddr, bramY);
		bramAddrFetchQ.deq;
		bramAddrCalcQ.enq(bramAddr);
	endrule
	
	
	rule fetchActivate0(fetchQ.first == ACTIVATE0);
		fetchQ.deq;
		//request BRAM values
		Bit#(9) bramAddrC = bramAddrFetchQ.first; //Carry
		Bit#(9) bramAddrF = bramAddrC*4+1; //F Layer
		bramReqA(bramAddrF, bramY);
		bramReqA(bramAddrC, bramC);
		bramAddrFetchQ.deq;
		bramAddrCalcQ.enq(bramAddrC);
	endrule
	
	rule fetchActivate1(fetchQ.first == ACTIVATE1);
		fetchQ.deq;
		//request BRAM values
		Bit#(9) bramAddr = bramAddrFetchQ.first;
		Bit#(9) bramAddrI = bramAddr*4;//I Layer
		Bit#(9) bramAddrC = bramAddr+2;//C Layer
		bramReqA(bramAddrI, bramY);
		bramReqB(bramAddrC, bramY);
		bramAddrFetchQ.deq;
		bramAddrCalcQ.enq(bramAddr);
	endrule
	
	rule fetchActivate2(fetchQ.first == ACTIVATE2);
		fetchQ.deq;
		//request BRAM values
		Bit#(9) bramAddr = bramAddrFetchQ.first;
		bramReqA(bramAddr, bramH); //carry*f
		bramReqA(bramAddr, bramC); //i*c
		bramAddrFetchQ.deq;
		bramAddrCalcQ.enq(bramAddr);
	endrule
	
	rule fetchActivate3(fetchQ.first == ACTIVATE3);
		fetchQ.deq;
		//request BRAM values
		Bit#(9) bramAddr = bramAddrFetchQ.first;
		bramReqA(bramAddr, bramC);
		bramAddrFetchQ.deq;
		bramAddrCalcQ.enq(bramAddr);
	endrule
	
	rule fetchActivate4(fetchQ.first == ACTIVATE4);
		fetchQ.deq;
		//request BRAM values
		Bit#(9) bramAddr = bramAddrFetchQ.first;
		Bit#(9) bramAddr0 = bramAddr*4+3;
		bramReqA(bramAddr, bramH);
		bramReqA(bramAddr, bramY);
		bramAddrFetchQ.deq;
		bramAddrCalcQ.enq(bramAddr);
	endrule
	
	function Int#(8) bramRespA(BRAM_DUAL_PORT#(Bit#(9), Bit#(8)) bram_port);
		return unpack(bram_port.a.read);
	endfunction
	
	function Int#(8) bramRespB(BRAM_DUAL_PORT#(Bit#(9), Bit#(8)) bram_port);
		return unpack(bram_port.b.read);
	endfunction
	
	function Int#(8) spramSelect(Bit#(16) spramValuePair, MaskHalf mask);
		Int#(8) value = 0;
		case (mask) matches
			UPPER: value = unpack(spramValuePair[15:8]);
			LOWER: value = unpack(spramValuePair[7:0]);
		endcase
		return value;
	endfunction	
	
	function Action bramWrite(Bit#(9) bram_addr, Bit#(8) value, BRAM_DUAL_PORT#(Bit#(9), Bit#(8)) bram_port);
		action
			bram_port.b.put(True, bram_addr, value);
		endaction
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


//main calculations
	rule calcInput(calcQ.first == INPUT);
		calcQ.deq;
		//Set aggregate value
		Int#(8) aggregateValue = 0;
		let readX = readQ.first;
        readQ.deq;
		if (readX) begin
			aggregateValue = bramRespA(bramX);
		end
		
		//receive spram kernel value
		Bit#(16) valuepair <- spram0.resp;
		let mask = maskCalcQ.first;
		maskCalcQ.deq;
		Int#(8) coeffValue = spramSelect(valuepair, mask);

		//get write address for BRAM X
		Bit#(9) bramAddrX = bramAddrCalcQ.first;
		bramAddrCalcQ.deq;
		//receive input value
		let inputValue = inputQ.first;
		if (bramAddrX == 399) begin
			inputQ.deq;
		end
		//Perform calculation and write to bram X
		Int#(8) product <- quantizedMult(coeffValue, inputValue);
		bramWrite(bramAddrX, pack(quantizedAdd(aggregateValue, product)), bramX);
	endrule
	
	rule calcHidden(calcQ.first == HIDDEN);
		calcQ.deq;
		//Set aggregate value
		Int#(8) aggregateValue = 0;
		let readY = readQ.first;
		readQ.deq;
		if (readY) begin
			aggregateValue = bramRespA(bramY);
		end
		//Set hidden value
		let hiddenValue = bramRespA(bramH);
		if (!readHiddenQ.first) hiddenValue = 0;
		readHiddenQ.deq;
		//receive spram recurrent value
		Bit#(16) valuepair = 0;
		case (spramTopQ.first) matches
			0: valuepair <- spram0.resp;
			1: valuepair <- spram1.resp;
		endcase
		spramTopQ.deq;
		let mask = maskCalcQ.first;
		maskCalcQ.deq;
		Int#(8) coeffValue = spramSelect(valuepair, mask);
		
		//get write address for BRAM Y
		Bit#(9) bramAddrY = bramAddrCalcQ.first;
		bramAddrCalcQ.deq;

		//Perform calculation and write to bram X
		Int#(8) product <- quantizedMult(coeffValue, hiddenValue);
		bramWrite(bramAddrY, pack(quantizedAdd(aggregateValue, product)), bramY);
	endrule
	
	function ActionValue#(Int#(8)) hardSigmoid(Int#(8) d)
		=
		actionvalue
			Int#(8) lowerlimit = -1;//-2.5f
			Int#(8) upperlimit = 1;//2.5f
			Int#(8) onepoint = 102; //1.0f
			Int#(8) alpha = 20; //0.2f
			Int#(8) offset = 51;//0.5f
			if (d <= lowerlimit) return truncate(zeropoint);
			else if (d >= upperlimit) return onepoint;
			else begin
				let m <- quantizedMult(alpha, d);
				return quantizedAdd(m, offset);
			end
		endactionvalue;
	
	FIFOF#(Int#(8)) bias1Q <- mkSizedBRAMFIFOF(2);
	
	rule calcBias(calcQ.first == BIAS);
		calcQ.deq;
		// Receive x value
		let xValue = bramRespA(bramX);
		// Receive y value
		let yValue = bramRespA(bramY);
		// Receive spram bias value
		Bit#(16) valuepair = 0;
		valuepair <- spram1.resp;
		let mask = maskCalcQ.first;
		maskCalcQ.deq;
		Int#(8) biasValue = spramSelect(valuepair, mask);
		//Perform calculation and send to next step
		bias1Q.enq(quantizedAdd(quantizedAdd(xValue,yValue), biasValue));
	endrule
	
	rule calcBias1(bias1Q.notEmpty);
		bias1Q.deq;
		let bValue = bias1Q.first;
		//get write address for BRAM Y
		Bit#(9) bramAddrY = bramAddrCalcQ.first;
		bramAddrCalcQ.deq;
		//Perform calculation and write to bram X
		let hsValue <- hardSigmoid(bValue);
		bramWrite(bramAddrY, pack(hsValue), bramY);
	endrule
	
//activate calculations
	rule calcActivate0(calcQ.first == ACTIVATE0);
		calcQ.deq;
		//Recieve carry value
		let carryValue = bramRespA(bramC);
		//Receive f value
		let fValue = bramRespA(bramY);
		//get write address for BRAM C
		Bit#(9) bramAddrC = bramAddrCalcQ.first;
		bramAddrCalcQ.deq;
		let multValue <- quantizedMult(carryValue, fValue);
		bramWrite(bramAddrC, pack(multValue), bramC);
	endrule
	
	rule calcActivate1(calcQ.first == ACTIVATE1);
		calcQ.deq;
		//Recieve i value
		let iValue = bramRespA(bramY);
		//Receive c value
		let cValue = bramRespB(bramY);
		//get write address for BRAM H
		Bit#(9) bramAddrH = bramAddrCalcQ.first;
		bramAddrCalcQ.deq;
		//Perform calculation and write to bram H
		let multValue <- quantizedMult(iValue, cValue);
		bramWrite(bramAddrH, pack(multValue), bramH);
	endrule
	
	rule calcActivate2(calcQ.first == ACTIVATE2);
		calcQ.deq;
		//Recieve i value
		let fCarryValue = bramRespA(bramC);
		//Receive c value
		let icValue = bramRespA(bramH);
		//get write address for BRAM H
		Bit#(9) bramAddrC = bramAddrCalcQ.first;
		bramAddrCalcQ.deq;
		//Perform calculation and write to bram C
		let cState = quantizedAdd(fCarryValue, icValue);
		bramWrite(bramAddrC, pack(cState), bramC);
	endrule	

	rule calcActivate3(calcQ.first == ACTIVATE3);
		calcQ.deq;
		//Recieve carry state 
		let carryState = bramRespA(bramC);
		//get write address for BRAM H
		Bit#(9) bramAddrH = bramAddrCalcQ.first;
		bramAddrCalcQ.deq;
		//Perform calculation and write to bram H
		let hsValue <- hardSigmoid(carryState);
		bramWrite(bramAddrH, pack(hsValue), bramH);
	endrule	
	
	rule calcActivate4(calcQ.first == ACTIVATE4);
		calcQ.deq;
		//Recieve hs value
		let hsValue = bramRespA(bramH);
		//Receive o value
		let oValue = bramRespA(bramY);
		//get write address for BRAM H
		Bit#(9) bramAddrH = bramAddrCalcQ.first;
		bramAddrCalcQ.deq;
		//Perform calculation and write to bram C
		let hState <- quantizedMult(hsValue, oValue);
		bramWrite(bramAddrH, pack(hState), bramH);
		lstm2.processInput(pack(hState));
	endrule

	/*
	rule mainResp(state==ACTIVATE);
		Int#(8) out <- lstm2.getOutput;
		outputQ.enq(out);
	endrule
	*/
	
	FIFOF#(Bit#(8)) weightQ <- mkSizedBRAMFIFOF(1);
	Reg#(Bit#(15)) spramAddrW <- mkReg(0);//Range 0-25199. Exclusively used by handle_weight
	Reg#(MaskHalf) writeMask <- mkReg(UPPER);//Exclusively used by handle_weight
	
//input processing
	method Action processInput(Bit#(8) weight);
		inputQ.enq(unpack(weight));
    endmethod
	
//weight processing
	method Action processWeight(Bit#(8) weight);
        Bit#(4) bytemask = 0;
		Bit#(16) maskeddata = 0;
		case (writeMask) matches //alternate the bytemask
			UPPER: begin
				writeMask <= LOWER;
				bytemask = 4'b1100;
				maskeddata[15:8] = weight;
			end
			LOWER: begin
				writeMask <= UPPER;
				bytemask = 4'b0011;
				if (spramAddrW < 25199) begin
                    spramAddrW <= spramAddrW + 1;
				end else begin //enable main
                    spramAddrW <= 0;
                    state <= MAIN; 
                end
				maskeddata[7:0] = weight;
			end
		endcase
		case (spramAddrW[14]) matches
				0: spram0.req(spramAddrW[13:0], maskeddata, True, bytemask);
				1: spram1.req(spramAddrW[13:0], maskeddata, True, bytemask);
		endcase
		$display("Writing LSTM1 weight spramAddr ", spramAddrW , " mask ", bytemask, " data ", maskeddata);
    endmethod
	
	method Action processLSTM2Weight(Bit#(8) weight);
		lstm2.processWeight(weight);
	endmethod
	
	method Action processDenseWeight(Bit#(8) weight);
		lstm2.processDenseWeight(weight);
	endmethod
	

	
//input processing
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
