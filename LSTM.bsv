import BRAMCore::*;
import FIFOF::*;

//invscale = 102;
//zeropoint = 0;

//lowerlimit = -1;//-2.5f
//upperlimit = 1;//2.5f
//onepoint = 102; //1.0f
//alpha = 20; //0.2f
//offset = 51;//0.5f


typedef enum {A, B} Port deriving(Bits,Eq);

typedef enum {INIT, INPUT, HIDDEN, BIAS, FIN} Stage deriving(Bits,Eq);

typedef enum {INIT, CALC, READ, EXPORT} State deriving(Bits,Eq);

interface LSTMIfc#(numeric type height, numeric type in_width, numeric type hidden_width, numeric type unit, numeric type offset);
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

module mkLSTM(LSTMIfc#(height, in_width, hidden_width, unit, offset));

	Reg#(Bit#(9)) input_counter <- mkReg(0);
	BRAM_DUAL_PORT#(Bit#(8), Bit#(16)) bram_x_if <- mkBRAMCore2(2* fromInteger(valueOf(unit)), False);
	BRAM_DUAL_PORT#(Bit#(8), Bit#(16)) bram_x_co <- mkBRAMCore2(2* fromInteger(valueOf(unit)), False);
	BRAM_DUAL_PORT#(Bit#(8), Bit#(16)) bram_y_if <- mkBRAMCore2(2* fromInteger(valueOf(unit)), False);
	BRAM_DUAL_PORT#(Bit#(8), Bit#(16)) bram_y_co <- mkBRAMCore2(2* fromInteger(valueOf(unit)), False);
	BRAM_DUAL_PORT#(Bit#(9), Bit#(8)) bram_hidden <- mkBRAMCore2(fromInteger(valueOf(unit)), False);
	BRAM_DUAL_PORT#(Bit#(9), Bit#(8)) bram_carry <- mkBRAMCore2(fromInteger(valueOf(unit)), False);
	
	Reg#(Bit#(14)) height_counter <- mkReg(0);//50
	Reg#(Bit#(14)) width_counter <- mkReg(0);//25 for input 100 for hidden
	Reg#(Bit#(14)) unit_counter <- mkReg(0);//100 / halved because of memory access
	
	Reg#(Bit#(8)) lstm_height_counter <- mkReg(0);//50
	Reg#(Bit#(8)) lstm_width_counter <- mkReg(0);//25 for input 100 for hidden
	Reg#(Bit#(8)) lstm_unit_counter <- mkReg(0);//100 / halved because of memory access

	Reg#(Stage) lstmStage <- mkReg(INIT);
	
	Reg#(Stage) reqStage <- mkReg(INIT);
	
	Reg#(Bool) spramReqReady <- mkReg(True);
	
	Reg#(State) state <- mkReg(INIT);
	
	Int#(16) invscale = 102;
	Int#(16) zeropoint = 0;

	FIFOF#(Bit#(14)) reqQ <- mkSizedFIFOF(2);
	FIFOF#(Int#(8)) inputQ <- mkSizedFIFOF(2);
	FIFOF#(Bit#(64)) spramQ <- mkSizedFIFOF(2);
	FIFOF#(Int#(8)) outputQ <- mkSizedFIFOF(2);
	
	
	rule reqRAM(reqStage != INIT && reqStage != FIN && reqQ.notFull);
		Bit#(14) unit = fromInteger(valueOf(unit));
		Bit#(14) in_width = fromInteger(valueOf(in_width));
		Bit#(14) hidden_width = fromInteger(valueOf(hidden_width));
		Bit#(14) height = fromInteger(valueOf(height));
		Bit#(14) offset = fromInteger(valueOf(offset));
		if (reqStage == INPUT) begin
			Bit#(14) req_addr = offset + width_counter*unit + unit_counter;
			reqQ.enq(req_addr);
			if (unit_counter < unit - 2) unit_counter <= unit_counter + 2;//extracting two at a time.
			else begin 
				unit_counter <= 0;
				if (width_counter < fromInteger(valueOf(in_width)) - 1) width_counter <= width_counter + 1;
				else begin
					width_counter <= 0;
					reqStage <= HIDDEN;
				end
			end
		end
		else if (reqStage == HIDDEN) begin
			Bit#(14) req_addr = offset + (unit*in_width/2) + width_counter*unit + unit_counter;
			reqQ.enq(req_addr);
			if (unit_counter < unit - 2) unit_counter <= unit_counter + 2;
			else begin 
				unit_counter <= 0;
				if (width_counter < hidden_width - 1) width_counter <= width_counter + 1;
				else begin
					width_counter <= 0;
					reqStage <= BIAS;
					
				end
			end
		end
		else if (reqStage == BIAS) begin
			Bit#(14) req_addr = offset + (unit*in_width/2) + (unit*hidden_width/2) +  width_counter*100 + unit_counter;
			reqQ.enq(req_addr);
			if (unit_counter < unit - 2) unit_counter <= unit_counter + 2;
			else begin
				if (height_counter < height) begin
					height_counter <= height_counter + 1;
					reqStage <= INPUT;
				end
				else begin
					height_counter <= 0;
					reqStage <= FIN;
				end
			end
		end	
		
		$display("req check");
		spramReqReady <= False;
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
	
	rule calculate1(state == CALC && lstmStage != FIN && spramQ.notEmpty);
		
		$display("calculating \n");
		Bit#(64) weights = spramQ.first;
		
		spramQ.deq;
		
		Int#(8) wt_a_i = unpack(weights[63:56]);
		Int#(8) wt_a_f = unpack(weights[47:40]);
		Int#(8) wt_a_c = unpack(weights[31:24]);
		Int#(8) wt_a_o = unpack(weights[15:8]);
		
		Int#(8) wt_b_i = unpack(weights[55:48]);
		Int#(8) wt_b_f = unpack(weights[39:32]);
		Int#(8) wt_b_c = unpack(weights[23:16]);
		Int#(8) wt_b_o = unpack(weights[7:0]);
		
		if (lstmStage == INPUT) begin
			
			//spramQ.deq;
			Int#(8) dataIn = inputQ.first;
			if (unit_counter == fromInteger(valueOf(unit)) - 2) inputQ.deq;
			
			Int#(8) x_a_i = 0;
			Int#(8) x_a_f = 0;
			Int#(8) x_a_c = 0;
			Int#(8) x_a_o = 0;
			Int#(8) x_b_i = 0;
			Int#(8) x_b_f = 0;
			Int#(8) x_b_c = 0;
			Int#(8) x_b_o = 0;
			
			if (height_counter > 0) begin
				let x_a_if = bram_x_if.a.read;
				let x_a_co = bram_x_co.a.read;
				
				let x_b_if = bram_x_if.b.read;
				let x_b_co = bram_x_co.b.read;
				
				x_a_i = unpack(x_a_if[15:8]);
				x_a_f = unpack(x_a_if[7:0]); 
				x_a_c = unpack(x_a_co[15:8]);
				x_a_o = unpack(x_a_co[7:0]); 
				
				x_b_i = unpack(x_b_if[15:8]);
				x_b_f = unpack(x_b_if[7:0]); 
				x_b_c = unpack(x_b_co[15:8]);
				x_b_o = unpack(x_b_co[7:0]); 
			end
			
			let p_a_i = quantizedMult(dataIn, wt_a_i);
			let p_a_f = quantizedMult(dataIn, wt_a_f);
			let p_a_c = quantizedMult(dataIn, wt_a_c);
			let p_a_o = quantizedMult(dataIn, wt_a_o);		
			
			let s_a_i = quantizedAdd(x_a_i, p_a_i);
			let s_a_f = quantizedAdd(x_a_f, p_a_f);
			let s_a_c = quantizedAdd(x_a_c, p_a_c);
			let s_a_o = quantizedAdd(x_a_o, p_a_o);	
			
			let p_b_i = quantizedMult(dataIn, wt_b_i);
			let p_b_f = quantizedMult(dataIn, wt_b_f);
			let p_b_c = quantizedMult(dataIn, wt_b_c);
			let p_b_o = quantizedMult(dataIn, wt_b_o);		
			
			let s_b_i = quantizedAdd(x_b_i, p_b_i);
			let s_b_f = quantizedAdd(x_b_f, p_b_f);
			let s_b_c = quantizedAdd(x_b_c, p_b_c);
			let s_b_o = quantizedAdd(x_b_o, p_b_o);
			
			Bit#(16) a_if;
			Bit#(16) a_co;
			Bit#(16) b_if;
			Bit#(16) b_co;
			
			a_if[15:8] = pack(s_a_i);
			a_if[7:0] = pack(s_a_f);
			a_co[15:8] = pack(s_a_c);
			a_co[7:0] = pack(s_a_o);
			
			b_if[15:8] = pack(s_b_i);
			b_if[7:0] = pack(s_b_f);
			b_co[15:8] = pack(s_b_c);
			b_co[7:0] = pack(s_b_o);
			
			bram_x_if.a.put(True, lstm_unit_counter, a_if);
			bram_x_co.a.put(True, lstm_unit_counter, a_co);
			bram_x_if.b.put(True, lstm_unit_counter+1, b_if);
			bram_x_co.b.put(True, lstm_unit_counter+1, b_co);
			
			lstm_unit_counter <= lstm_unit_counter + 2;
			
		end
		else if (lstmStage == HIDDEN) begin
			
			Int#(8) h_prev = 0;
			if (lstm_width_counter > 0) h_prev = unpack(bram_hidden.a.read);
			
			Int#(8) y_a_i = 0;
			Int#(8) y_a_f = 0;
			Int#(8) y_a_c = 0;
			Int#(8) y_a_o = 0;
			Int#(8) y_b_i = 0;
			Int#(8) y_b_f = 0;
			Int#(8) y_b_c = 0;
			Int#(8) y_b_o = 0;
			
			if (height_counter > 0) begin
				let y_a_if = bram_y_if.a.read;
				let y_a_co = bram_y_co.a.read;
				
				let y_b_if = bram_y_if.b.read;
				let y_b_co = bram_y_co.b.read;
				
				y_a_i = unpack(y_a_if[15:8]);
				y_a_f = unpack(y_a_if[7:0]); 
				y_a_c = unpack(y_a_co[15:8]);
				y_a_o = unpack(y_a_co[7:0]); 
				
				y_b_i = unpack(y_b_if[15:8]);
				y_b_f = unpack(y_b_if[7:0]); 
				y_b_c = unpack(y_b_co[15:8]);
				y_b_o = unpack(y_b_co[7:0]); 
			end
			
			let p_a_i = quantizedMult(h_prev, wt_a_i);
			let p_a_f = quantizedMult(h_prev, wt_a_f);
			let p_a_c = quantizedMult(h_prev, wt_a_c);
			let p_a_o = quantizedMult(h_prev, wt_a_o);		
			
			let s_a_i = quantizedAdd(y_a_i, p_a_i);
			let s_a_f = quantizedAdd(y_a_f, p_a_f);
			let s_a_c = quantizedAdd(y_a_c, p_a_c);
			let s_a_o = quantizedAdd(y_a_o, p_a_o);	
			
			let p_b_i = quantizedMult(h_prev, wt_b_i);
			let p_b_f = quantizedMult(h_prev, wt_b_f);
			let p_b_c = quantizedMult(h_prev, wt_b_c);
			let p_b_o = quantizedMult(h_prev, wt_b_o);		
			
			let s_b_i = quantizedAdd(y_b_i, p_b_i);
			let s_b_f = quantizedAdd(y_b_f, p_b_f);
			let s_b_c = quantizedAdd(y_b_c, p_b_c);
			let s_b_o = quantizedAdd(y_b_o, p_b_o);
			
			Bit#(16) a_if;
			Bit#(16) a_co;
			Bit#(16) b_if;
			Bit#(16) b_co;
			
			a_if[15:8] = pack(s_a_i);
			a_if[7:0] = pack(s_a_f);
			a_co[15:8] = pack(s_a_c);
			a_co[7:0] = pack(s_a_o);
			
			b_if[15:8] = pack(s_b_i);
			b_if[7:0] = pack(s_b_f);
			b_co[15:8] = pack(s_b_c);
			b_co[7:0] = pack(s_b_o);
			
			bram_y_if.a.put(True, lstm_unit_counter, a_if);
			bram_y_co.a.put(True, lstm_unit_counter, a_co);
			bram_y_if.b.put(True, lstm_unit_counter+1, b_if);
			bram_y_co.b.put(True, lstm_unit_counter+1, b_co);
			
			lstm_unit_counter <= lstm_unit_counter + 2;
			
		end
		else if (lstmStage == BIAS) begin
			
			let x_a_if = bram_x_if.a.read;
			let x_a_co = bram_x_co.a.read;
			
			let x_b_if = bram_x_if.b.read;
			let x_b_co = bram_x_co.b.read;
			
			
			let y_a_if = bram_y_if.a.read;
			let y_a_co = bram_y_co.a.read;
				
			let y_b_if = bram_y_if.b.read;
			let y_b_co = bram_y_co.b.read;
			
			Int#(8) c_a_prev = 0;
			Int#(8) c_b_prev = 0;
			if (lstm_width_counter > 0) begin
				c_a_prev = unpack(bram_carry.a.read);
				c_b_prev = unpack(bram_carry.b.read);
			end

			Int#(8) x_a_i = unpack(x_a_if[15:8]);
			Int#(8) x_a_f = unpack(x_a_if[7:0]); 
			Int#(8) x_a_c = unpack(x_a_co[15:8]);
			Int#(8) x_a_o = unpack(x_a_co[7:0]); 
			Int#(8) x_b_i = unpack(x_b_if[15:8]);
			Int#(8) x_b_f = unpack(x_b_if[7:0]); 
			Int#(8) x_b_c = unpack(x_b_co[15:8]);
			Int#(8) x_b_o = unpack(x_b_co[7:0]); 
		
			Int#(8) y_a_i = unpack(y_a_if[15:8]);
			Int#(8) y_a_f = unpack(y_a_if[7:0]); 
			Int#(8) y_a_c = unpack(y_a_co[15:8]);
			Int#(8) y_a_o = unpack(y_a_co[7:0]); 
			Int#(8) y_b_i = unpack(y_b_if[15:8]);
			Int#(8) y_b_f = unpack(y_b_if[7:0]); 
			Int#(8) y_b_c = unpack(y_b_co[15:8]);
			Int#(8) y_b_o = unpack(y_b_co[7:0]); 
			
			let sy_a_i = hardSigmoid(quantizedAdd(quantizedAdd(y_a_i, x_a_i), wt_a_i));
			let sy_a_f = hardSigmoid(quantizedAdd(quantizedAdd(y_a_f, x_a_f), wt_a_f));
			let sy_a_c = hardSigmoid(quantizedAdd(quantizedAdd(y_a_c, x_a_c), wt_a_c));
			let sy_a_o = hardSigmoid(quantizedAdd(quantizedAdd(y_a_o, x_a_o), wt_a_o));
			
			let sy_a_ifc = quantizedAdd(quantizedMult(sy_a_f, c_a_prev), quantizedMult(sy_a_i, sy_a_c)); //goes in the carry
			let sy_a_ifco = quantizedMult(sy_a_o, hardSigmoid(sy_a_ifc)); //goes in hidden
			
			let sy_b_i = hardSigmoid(quantizedAdd(quantizedAdd(y_b_i, x_b_i), wt_b_i));
			let sy_b_f = hardSigmoid(quantizedAdd(quantizedAdd(y_b_f, x_b_f), wt_b_f));
			let sy_b_c = hardSigmoid(quantizedAdd(quantizedAdd(y_b_c, x_b_c), wt_b_c));
			let sy_b_o = hardSigmoid(quantizedAdd(quantizedAdd(y_b_o, x_b_o), wt_b_o));
			
			let sy_b_ifc = quantizedAdd(quantizedMult(sy_b_f, c_b_prev), quantizedMult(sy_b_i, sy_b_c)); //goes in the carry
			let sy_b_ifco = quantizedMult(sy_b_o, hardSigmoid(sy_b_ifc)); //goes in hidden
			
			Bit#(9) addr = extend( lstm_unit_counter);
			bram_carry.a.put(True,addr, pack(sy_a_ifc));
			bram_hidden.a.put(True, addr, pack(sy_a_ifco));
			
			bram_carry.b.put(True, addr+1, pack(sy_b_ifc));
			bram_hidden.b.put(True, addr+1, pack(sy_b_ifco));
			
			lstm_unit_counter <= lstm_unit_counter + 2;
		end
		state  <= READ;
	endrule
	
	rule calculate2(state == READ ); //set up bram read requests
	
		$display("calculating2 \n");
		Bit#(8) nextCounter = lstm_unit_counter;
		Bit#(8) archCounter = lstm_width_counter;
		Bit#(8) totalCounter = lstm_height_counter;
		Stage nextStage = lstmStage; 
		State nextState = CALC;
		if (nextCounter == fromInteger(valueOf(unit))) begin
			if (lstmStage == INPUT) begin
				if (archCounter == fromInteger(valueOf(in_width)) - 1) begin
					nextStage = HIDDEN;
					archCounter = 0;
				end else begin
					archCounter = archCounter + 1;
				end
				nextCounter = 0;
			end
			else if (lstmStage == HIDDEN) begin
				if (archCounter == fromInteger(valueOf(hidden_width)) - 1) begin
					nextStage = BIAS;
					archCounter = 0;
				end else begin
					archCounter = archCounter + 1;
				end
				nextCounter = 0;	
			end
			else if (lstmStage == BIAS) begin
				if (totalCounter == fromInteger(valueOf(height)) - 1) begin
					nextStage = FIN;
				end else begin
					nextState = EXPORT;
					totalCounter = totalCounter + 1;
					$display("Iteration: %u", lstm_height_counter);
					bram_hidden.a.put(False, 0, ?);
				end
				nextCounter = 0;
			end 
		end
		Bit#(8) nextNext = nextCounter+1;
		if (nextStage == INPUT) begin //up to 200
		
			bram_x_if.a.put(False, nextCounter, ?);
			bram_x_co.a.put(False, nextCounter, ?);
			
			bram_x_if.b.put(False, nextNext, ?);
			bram_x_co.b.put(False, nextNext, ?);		
		end
		else if (nextStage == HIDDEN) begin // up to 200
			bram_y_if.a.put(False, nextCounter, ?);
			bram_y_co.a.put(False, nextCounter, ?);
			
			bram_y_if.b.put(False, nextNext, ?);
			bram_y_co.b.put(False, nextNext, ?);
			
			if (nextCounter == 0) begin
				bram_hidden.a.put(False, extend(archCounter), ?);
			end
		end
		else if (lstmStage == BIAS) begin
			bram_y_if.a.put(False, nextCounter, ?);
			bram_y_co.a.put(False, nextCounter, ?);
				
			bram_y_if.b.put(False, nextNext, ?);
			bram_y_co.b.put(False, nextNext, ?);
			
			Bit#(9) addr = extend(nextCounter);
			
			bram_carry.a.put(False, addr, ?);
			bram_carry.b.put(False, addr+1, ?);
		end
		
		lstm_unit_counter <=  nextCounter;
		lstm_width_counter <= archCounter;
		lstm_height_counter <= totalCounter;
		state <= nextState;
	endrule
	
	rule calculate3(state == EXPORT);
		
		let hidden_a = unpack(bram_hidden.a.read);
		outputQ.enq(hidden_a);
		if (unit_counter < fromInteger(valueOf(unit))) begin
			bram_hidden.a.put(False, 0, ?);
			unit_counter <= unit_counter + 1;
		end else begin
			state <= CALC;
			unit_counter <= 0;
		end
	endrule
	
	method Action processWeight(Bit#(64) weights);
		spramQ.enq(weights);
	endmethod
	
	method Action processInput(Int#(8) in);
		inputQ.enq(in);
	endmethod
		
	method Action start;
		if (reqStage == INIT) begin
			lstmStage <= INPUT;
			reqStage <= INPUT;
			state <= CALC;
			$display("start check");
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
	
	
endmodule: mkLSTM