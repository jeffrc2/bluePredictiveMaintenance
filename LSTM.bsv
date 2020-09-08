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

typedef enum {INIT, CALC, FIN} State deriving(Bits,Eq);

typedef enum {INPUT1, INPUT2, INPUTFETCH, HIDDEN1, HIDDEN2, HIDDENFETCH, BIAS1, BIAS2, BIAS3, BIAS4, BIAS5, BIAS6, BIAS7, BIASFETCH, FIN} CalcStage deriving(Bits,Eq);

interface LSTMIfc#(numeric type height, numeric type in_width, numeric type hidden_width, numeric type unit, numeric type offset);
	method Action processInput(Int#(8) in); //Put the input into the input queue.
	method Action processWeight(Bit#(64) weights); //Put the weights into the weights queue.
	method Action start; 
	method ActionValue#(Bit#(14)) getRequest; //retrieves request in the request queue.
	method Bool inputReady;
	method Bool requestQueued;
	method Bool outputQueued;
	method ActionValue#(Int#(8)) getOutput;
endinterface

module mkLSTM(LSTMIfc#(height, in_width, hidden_width, unit, offset));

	//QuantArithIfc#(height, in_width, hidden_width, unit, offset) quantArith <- mkQuantArith;

	Reg#(Bit#(14)) input_counter <- mkReg(0);
	
	Reg#(Bit#(14)) output_counter <- mkReg(0);

	Reg#(Bit#(14)) req_input_counter <- mkReg(0);
	Reg#(Bit#(14)) req_height_counter <- mkReg(0);//50 for LSTM1&2
	Reg#(Bit#(14)) req_width_counter <- mkReg(0);//25 for input 100 for hidden
	Reg#(Bit#(14)) req_unit_counter <- mkReg(0);//100 / halved because of memory access
	
	Reg#(Bit#(8)) lstm_input_counter <- mkReg(0);
	Reg#(Bit#(8)) lstm_height_counter <- mkReg(0);//50 for LSTM1&2
	Reg#(Bit#(8)) lstm_width_counter <- mkReg(0);//25 for input 100 for hidden
	Reg#(Bit#(8)) lstm_unit_counter <- mkReg(0);//100 / halved because of memory access
	
	Reg#(Bit#(64)) input_calc1_x <- mkReg(0);
	Reg#(Bit#(64)) input_calc1_p <- mkReg(0);
	
	Reg#(Bit#(64)) hidden_calc1_y <- mkReg(0);
	Reg#(Bit#(64)) hidden_calc1_p <- mkReg(0);

	
	Reg#(Bit#(32)) bias_calc1_a <- mkReg(0);
	Reg#(Bit#(32)) bias_calc1_b <- mkReg(0);
	Reg#(Bit#(64)) bias_calc1_wt <- mkReg(0);
	Reg#(Bit#(16)) bias_calc1_c <- mkReg(0);

	Reg#(Bit#(64)) bias_calc2_s <- mkReg(0);
	Reg#(Bit#(16)) bias_calc2_c <- mkReg(0);
	
	Reg#(Bit#(64)) bias_calc3_hs <- mkReg(0);
	Reg#(Bit#(16)) bias_calc3_c <- mkReg(0);
	
	Reg#(Bit#(16)) bias_calc4_o <- mkReg(0);
	Reg#(Bit#(16)) bias_calc4_ic <- mkReg(0);
	Reg#(Bit#(16)) bias_calc4_fprev <- mkReg(0);
	
	Reg#(Bit#(16)) bias_calc5_o <- mkReg(0);
	Reg#(Bit#(16)) bias_calc5_ifc <- mkReg(0);
	
	Reg#(Bit#(16)) bias_calc6_hs_ifc <- mkReg(0);
	Reg#(Bit#(16)) bias_calc6_o <- mkReg(0);
	Reg#(Bit#(16)) bias_calc6_ifc <- mkReg(0);
	
	Reg#(Stage) reqStage <- mkReg(INIT);
	
	Reg#(CalcStage) calcStage <- mkReg(INPUT1);
	
	Reg#(Stage) lstmStage <- mkReg(INIT);
	
	Reg#(State) state <- mkReg(INIT);

	FIFOF#(Bit#(14)) reqQ <- mkSizedFIFOF(2);
	FIFOF#(Int#(8)) inputQ <- mkSizedFIFOF(2);
	FIFOF#(Bit#(64)) spramQ <- mkSizedFIFOF(2);
	FIFOF#(Int#(8)) outputQ <- mkSizedFIFOF(2);
		
	Int#(16) invscale = 102;
	Int#(16) zeropoint = 0;
	
	BRAM_DUAL_PORT#(Bit#(8), Bit#(16)) bram_x_if <- mkBRAMCore2(2* fromInteger(valueOf(unit)), False);
	BRAM_DUAL_PORT#(Bit#(8), Bit#(16)) bram_x_co <- mkBRAMCore2(2* fromInteger(valueOf(unit)), False);
	BRAM_DUAL_PORT#(Bit#(8), Bit#(16)) bram_y_if <- mkBRAMCore2(2* fromInteger(valueOf(unit)), False);
	BRAM_DUAL_PORT#(Bit#(8), Bit#(16)) bram_y_co <- mkBRAMCore2(2* fromInteger(valueOf(unit)), False);
	BRAM_DUAL_PORT#(Bit#(9), Bit#(8)) bram_hidden <- mkBRAMCore2(fromInteger(valueOf(unit)), False);
	BRAM_DUAL_PORT#(Bit#(9), Bit#(8)) bram_carry <- mkBRAMCore2(fromInteger(valueOf(unit)), False);
	
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
	
	
	//Rule requesting the next SPRAM weights when the request queue is empty
	rule reqRAM(reqStage != INIT && reqStage != FIN && !reqQ.notEmpty);
		`ifdef BSIM
		$display("reqRAM");
		`endif
		
		Bit#(14) unit = fromInteger(valueOf(unit));
		Bit#(14) in_width = fromInteger(valueOf(in_width));
		Bit#(14) hidden_width = fromInteger(valueOf(hidden_width));
		Bit#(14) height = fromInteger(valueOf(height));
		Bit#(14) offset = fromInteger(valueOf(offset));
		
		//For 50 time steps
			//Reset stored x and y state
			//For 100 units
				
				//For 25 input sequences
					//Request kernel weights for state accumulation with input
				//For 100 hidden sequences
					//Request weights for state accumulation with hidden state
			//For 100 units
				//Request weights for layer sigmoid processing.
		Bit#(14) req_addr = 0;
		
		case (reqStage) matches 
			INPUT: begin
				req_addr = offset + (req_input_counter*unit/2) + req_unit_counter/2;
				if (req_unit_counter == unit - 2) begin
					req_unit_counter <= 0;
					if (req_input_counter == in_width - 1) begin
						req_input_counter <= 0;
						reqStage <= HIDDEN;
					end else req_input_counter <= req_input_counter + 1;
				end else req_unit_counter <= req_unit_counter + 2;
				

			end
			HIDDEN: begin
				req_addr = offset + (unit*in_width/2) + (req_width_counter*unit/2) + req_unit_counter/2;
				if (req_unit_counter == unit - 2) begin
					req_unit_counter <= 0;
					if (req_width_counter == hidden_width - 1) begin
						req_width_counter <= 0;
						reqStage <= BIAS;
					end else req_width_counter <= req_width_counter + 1;
				end else req_unit_counter <= req_unit_counter + 2;
				
				
			end
			BIAS: begin
				Stage next = INPUT;
				req_addr = offset + (unit*in_width/2) + (unit*hidden_width/2) + (req_width_counter*unit/2) + req_unit_counter/2;
				if (req_unit_counter == unit - 2) begin
					next = INPUT;
					req_unit_counter <= 0;
					if (req_height_counter == height - 1) begin
						//done
						next = FIN;
						req_height_counter <= 0;
					end else req_height_counter <= req_height_counter + 1;
				end else req_unit_counter <= req_unit_counter + 2;
				reqStage <= next;
			end
		endcase
		reqQ.enq(req_addr);
	endrule

	rule fetch(calcStage == INPUTFETCH || calcStage == HIDDENFETCH || calcStage == BIASFETCH);
		
		Bit#(8) unit = fromInteger(valueOf(unit));
		Bit#(8) in_width = fromInteger(valueOf(in_width));
		Bit#(8) hidden_width = fromInteger(valueOf(hidden_width));
		Bit#(8) height = fromInteger(valueOf(height));
		Bit#(8) offset = fromInteger(valueOf(offset));
		
		case (calcStage) matches
			INPUTFETCH: begin 
				CalcStage next = INPUT1;
				Bit#(8) addrA = lstm_unit_counter + 2;
				Bit#(8) addrB = lstm_input_counter + 3;
				if (lstm_unit_counter == unit - 2) begin
				
					addrA = 0;
					addrB = 1;
					lstm_unit_counter <= 0;
					if (lstm_input_counter == in_width - 1) begin //Finished Input sequence set, switch to Hidden sublayer 
						lstm_input_counter <= 0;
						next = HIDDEN1;
						
						bram_y_if.a.put(False, addrA, ?);
						bram_y_co.a.put(False, addrA, ?);
						bram_y_if.b.put(False, addrB, ?);
						bram_y_co.b.put(False, addrB, ?);
						
						bram_hidden.a.put(False, zeroExtend(addrA), ?);
						
						inputQ.deq; //current sequence finished, move to next input	
					end else begin //Finished Input sequence
						lstm_input_counter <= lstm_input_counter + 1;
						
						bram_x_if.a.put(False, addrA, ?);
						bram_x_co.a.put(False, addrA, ?);
						bram_x_if.b.put(False, addrB, ?);
						bram_x_co.b.put(False, addrB, ?);
						
						
					end
				end else begin //Continue processing on existing Input sequence
					lstm_unit_counter <= lstm_unit_counter + 2;
					
					bram_x_if.a.put(False, addrA, ?);
					bram_x_co.a.put(False, addrA, ?);
					bram_x_if.b.put(False, addrB, ?);
					bram_x_co.b.put(False, addrB, ?);
				end
				calcStage <= next;
		
			end
			HIDDENFETCH: begin
				CalcStage next = HIDDEN1;
				
				Bit#(8) addrA = lstm_unit_counter + 2;
				Bit#(8) addrB = lstm_unit_counter + 3;
				if (lstm_unit_counter == unit - 2) begin
					
					addrA = 0;
					addrB = 1;
					lstm_unit_counter <= 0;
					if (lstm_width_counter == hidden_width - 1) begin  //Finished Hidden state set, switch to Bias sublayer 
						lstm_width_counter <= 0;
						next = BIAS1;

						bram_y_if.a.put(False, addrA, ?);
						bram_y_co.a.put(False, addrA, ?);
						bram_y_if.b.put(False, addrB, ?);
						bram_y_co.b.put(False, addrB, ?);
						
						bram_carry.a.put(False, zeroExtend(addrA), ?);
						bram_carry.b.put(False, zeroExtend(addrB), ?);
						
					end else begin //Current Hidden value processed, move to next Hidden value
					
						lstm_width_counter <= lstm_width_counter + 1;
						bram_hidden.a.put(False, signExtend(lstm_width_counter + 1), ?);
						
					end
				end else begin //Continue processing on existing Hidden value
				
					lstm_unit_counter <= lstm_unit_counter + 2;
					bram_y_if.a.put(False, addrA, ?);
					bram_y_co.a.put(False, addrA, ?);
					bram_y_if.b.put(False, addrB, ?);
					bram_y_co.b.put(False, addrB, ?);
					
				end
				calcStage <= next;
		
			end
			BIASFETCH: begin
				CalcStage next = HIDDEN1;
				
				Bit#(8) addrA = lstm_unit_counter + 2;
				Bit#(8) addrB = lstm_unit_counter + 3;
				
				if (lstm_unit_counter == unit - 2) begin 
					next = INPUT1;
					lstm_unit_counter <= 0;
					if (lstm_height_counter == height - 1) begin //Finished the entire timestep set
						//done
						next = FIN;
						lstm_height_counter <= 0;
						
					end else begin  //Finished with the current timestep, Move to the next time set
						lstm_height_counter <= lstm_height_counter + 1;
						
					end
				end else begin //Continue with the current timestep
					lstm_unit_counter <= lstm_unit_counter + 2;
					
					bram_x_if.a.put(False, addrA, ?);
					bram_x_co.a.put(False, addrA, ?);
					bram_x_if.b.put(False, addrB, ?);
					bram_x_co.b.put(False, addrB, ?);
					
					bram_y_if.a.put(False, addrA, ?);
					bram_y_co.a.put(False, addrA, ?);
					bram_y_if.b.put(False, addrB, ?);
					bram_y_co.b.put(False, addrB, ?);
					
					bram_carry.a.put(False, zeroExtend(addrA), ?);
					bram_carry.b.put(False, zeroExtend(addrB), ?);
					
				end
				calcStage <= next;
			end
		endcase
		
	endrule

	//Rule processing input and the the corresponding weights.
	rule calcInput1(state == CALC && calcStage == INPUT1 && spramQ.notEmpty && inputQ.notEmpty);
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
		
		Int#(8) dataIn = inputQ.first;
	
		Int#(8) x_zero = 0;
		Bit#(64) x = 0;
		
		if (lstm_input_counter > 0) begin
			let x_a_if = bram_x_if.a.read;
			let x_a_co = bram_x_co.a.read;
			
			let x_b_if = bram_x_if.b.read;
			let x_b_co = bram_x_co.b.read;
			
			x[63:48] = x_a_if;
			x[47:32] = x_a_co;
			x[31:16] = x_b_if;
			x[15:0] = x_b_co;
		end else begin
			x[63:56] = pack(x_zero);
			x[55:48] = pack(x_zero);
			x[47:40] = pack(x_zero);
			x[39:32] = pack(x_zero);
			
			x[31:24] = pack(x_zero);
			x[23:16] = pack(x_zero);
			x[15:8] = pack(x_zero);
			x[7:0] = pack(x_zero);
		end
		
		Bit#(64) p = 0;
		
		let p_a_i = quantizedMult(dataIn, wt_a_i);
		let p_a_f = quantizedMult(dataIn, wt_a_f);
		let p_a_c = quantizedMult(dataIn, wt_a_c);
		let p_a_o = quantizedMult(dataIn, wt_a_o);		
		
		let p_b_i = quantizedMult(dataIn, wt_b_i);
		let p_b_f = quantizedMult(dataIn, wt_b_f);
		let p_b_c = quantizedMult(dataIn, wt_b_c);
		let p_b_o = quantizedMult(dataIn, wt_b_o);	
		
		p[63:56] = pack(p_a_i);
		p[55:48] = pack(p_a_f);
		p[47:40] = pack(p_a_c);
		p[39:32] = pack(p_a_o);
		p[31:24] = pack(p_b_i);
		p[23:16] = pack(p_b_f);
		p[15:8] = pack(p_b_c);
		p[7:0] = pack(p_b_o);
		
		input_calc1_x <= x;
		input_calc1_p <= p;
	
		calcStage <= INPUT2;
	endrule
	
	rule calcInput2(calcStage == INPUT2);
		Bit#(64) x = input_calc1_x;
		Bit#(64) p = input_calc1_p;
		
		Int#(8) x_a_i = unpack(x[63:56]);
		Int#(8) x_a_f = unpack(x[55:48]); 
		Int#(8) x_a_c = unpack(x[47:40]);
		Int#(8) x_a_o = unpack(x[39:32]); 
		
		Int#(8) x_b_i = unpack(x[31:24]);
		Int#(8) x_b_f = unpack(x[23:16]); 
		Int#(8) x_b_c = unpack(x[15:8]);
		Int#(8) x_b_o = unpack(x[7:0]); 
		
		Int#(8) p_a_i = unpack(p[63:56]);
		Int#(8) p_a_f = unpack(p[55:48]);
		Int#(8) p_a_c = unpack(p[47:40]);
		Int#(8) p_a_o = unpack(p[39:32]);
		
		Int#(8) p_b_i = unpack(p[31:24]);
		Int#(8) p_b_f = unpack(p[23:16]);
		Int#(8) p_b_c = unpack(p[15:8]);
		Int#(8) p_b_o = unpack(p[7:0]);
	
		let s_a_i = quantizedAdd(x_a_i, p_a_i);
		let s_a_f = quantizedAdd(x_a_f, p_a_f);
		let s_a_c = quantizedAdd(x_a_c, p_a_c);
		let s_a_o = quantizedAdd(x_a_o, p_a_o);	
			
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
		
		calcStage <= INPUTFETCH;
	endrule
	
	rule calcHidden1(state == CALC && calcStage == HIDDEN1 && spramQ.notEmpty);
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
		
		Int#(8) y_zero = 0;
		
		Bit#(64) y = 0;
		
		if (lstm_unit_counter > 0) begin
			let y_a_if = bram_y_if.a.read;
			let y_a_co = bram_y_co.a.read;
			
			let y_b_if = bram_y_if.b.read;
			let y_b_co = bram_y_co.b.read;
			
			y[63:48] = y_a_if;
			y[47:32] = y_a_co;
			y[31:16] = y_b_if;
			y[15:0] = y_b_co;
		end else begin
			y[63:56] = pack(y_zero);
			y[55:48] = pack(y_zero);
			y[47:40] = pack(y_zero);
			y[39:32] = pack(y_zero);
			
			y[31:24] = pack(y_zero);
			y[23:16] = pack(y_zero);
			y[15:8] = pack(y_zero);
			y[7:0] = pack(y_zero);
		end
		
		Bit#(64) p = 0;
		
		let p_a_i = quantizedMult(h_prev, wt_a_i);
		let p_a_f = quantizedMult(h_prev, wt_a_f);
		let p_a_c = quantizedMult(h_prev, wt_a_c);
		let p_a_o = quantizedMult(h_prev, wt_a_o);		
		
		let p_b_i = quantizedMult(h_prev, wt_b_i);
		let p_b_f = quantizedMult(h_prev, wt_b_f);
		let p_b_c = quantizedMult(h_prev, wt_b_c);
		let p_b_o = quantizedMult(h_prev, wt_b_o);
		
		p[63:56] = pack(p_a_i);
		p[55:48] = pack(p_a_f);
		p[47:40] = pack(p_a_c);
		p[39:32] = pack(p_a_o);
		p[31:24] = pack(p_b_i);
		p[23:16] = pack(p_b_f);
		p[15:8] = pack(p_b_c);
		p[7:0] = pack(p_b_c);
		
		hidden_calc1_y <= y;
		hidden_calc1_p <= p;
		
		calcStage <= HIDDEN2;
	endrule
	
	rule calcHidden2(calcStage == HIDDEN2);
		Bit#(64) y = hidden_calc1_y;
		Bit#(64) p = hidden_calc1_p;
		
		Int#(8) y_a_i = unpack(y[63:56]);
		Int#(8) y_a_f = unpack(y[55:48]); 
		Int#(8) y_a_c = unpack(y[47:40]);
		Int#(8) y_a_o = unpack(y[39:32]); 
		
		Int#(8) y_b_i = unpack(y[31:24]);
		Int#(8) y_b_f = unpack(y[23:16]); 
		Int#(8) y_b_c = unpack(y[15:8]);
		Int#(8) y_b_o = unpack(y[7:0]); 
		
		Int#(8) p_a_i = unpack(p[63:56]);
		Int#(8) p_a_f = unpack(p[55:48]);
		Int#(8) p_a_c = unpack(p[47:40]);
		Int#(8) p_a_o = unpack(p[39:32]);
		
		Int#(8) p_b_i = unpack(p[31:24]);
		Int#(8) p_b_f = unpack(p[23:16]);
		Int#(8) p_b_c = unpack(p[15:8]);
		Int#(8) p_b_o = unpack(p[7:0]);
		
		let s_a_i = quantizedAdd(y_a_i, p_a_i);
		let s_a_f = quantizedAdd(y_a_f, p_a_f);
		let s_a_c = quantizedAdd(y_a_c, p_a_c);
		let s_a_o = quantizedAdd(y_a_o, p_a_o);	
		
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
		
		calcStage <= HIDDENFETCH;
	endrule
	
	rule calcBias1(state == CALC && calcStage == BIAS1 && spramQ.notEmpty);
		
		let x_a_if = bram_x_if.a.read;
		let x_a_co = bram_x_co.a.read;
		
		let x_b_if = bram_x_if.b.read;
		let x_b_co = bram_x_co.b.read;
		
		let y_a_if = bram_y_if.a.read;
		let y_a_co = bram_y_co.a.read;
			
		let y_b_if = bram_y_if.b.read;
		let y_b_co = bram_y_co.b.read;

		Bit#(16) c = 0;
		Int#(8) c_zero = 0;
		
		
		if (lstm_width_counter > 0) begin
			Int#(8) c_a_prev = unpack(bram_carry.a.read);
			Int#(8) c_b_prev = unpack(bram_carry.b.read);
			c[15:8] = pack(c_a_prev);
			c[7:0] = pack(c_b_prev);
		end else begin
			c[15:8] = pack(c_zero);
			c[7:0] = pack(c_zero);
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
		
		Bit#(32) a = 0;
		Bit#(32) b = 0;
		
		let a_i = quantizedAdd(y_a_i, x_a_i);
		let a_f = quantizedAdd(y_a_f, x_a_f);
		let a_c = quantizedAdd(y_a_c, x_a_c);
		let a_o = quantizedAdd(y_a_o, x_a_o);
		
		let b_i = quantizedAdd(y_b_i, x_b_i);
		let b_f = quantizedAdd(y_b_f, x_b_f);
		let b_c = quantizedAdd(y_b_c, x_b_c);
		let b_o = quantizedAdd(y_b_o, x_b_o);
		
		a[31:24] = pack(a_i);
		a[23:16] = pack(a_f);
		a[15:8] = pack(a_c);
		a[7:0] = pack(a_o);

		b[31:24] = pack(b_i);
		b[23:16] = pack(b_f);
		b[15:8] = pack(b_c);
		b[7:0] = pack(b_o);
		
		Bit#(64) weights = spramQ.first;
		spramQ.deq;
	
		bias_calc1_a <= a;
		bias_calc1_b <= b;
		bias_calc1_c <= c;
		bias_calc1_wt <= weights;
		
		calcStage <= BIAS2;
	endrule
	
	rule calcBias2(calcStage == BIAS2);
		Bit#(32) a = bias_calc1_a;
		Bit#(32) b = bias_calc1_b;
		Bit#(64) weights = bias_calc1_wt;
		
		Int#(8) a_i = unpack(a[31:24]);
		Int#(8) a_f = unpack(a[23:16]);
		Int#(8) a_c = unpack(a[15:8]);
		Int#(8) a_o = unpack(a[7:0]);
		
		Int#(8) b_i = unpack(b[31:24]);
		Int#(8) b_f = unpack(b[23:16]);
		Int#(8) b_c = unpack(b[15:8]);
		Int#(8) b_o = unpack(b[7:0]);
		
		Int#(8) wt_a_i = unpack(weights[63:56]);
		Int#(8) wt_a_f = unpack(weights[47:40]);
		Int#(8) wt_a_c = unpack(weights[31:24]);
		Int#(8) wt_a_o = unpack(weights[15:8]);
		
		Int#(8) wt_b_i = unpack(weights[55:48]);
		Int#(8) wt_b_f = unpack(weights[39:32]);
		Int#(8) wt_b_c = unpack(weights[23:16]);
		Int#(8) wt_b_o = unpack(weights[7:0]);
		
		Bit#(64) s = 0;
	
		let s_a_i = quantizedAdd(a_i, wt_a_i);
		let s_a_f = quantizedAdd(a_f, wt_a_f);
		let s_a_c = quantizedAdd(a_c, wt_a_c);
		let s_a_o = quantizedAdd(a_o, wt_a_o);
		
		let s_b_i = quantizedAdd(b_i, wt_b_i);
		let s_b_f = quantizedAdd(b_f, wt_b_f);
		let s_b_c = quantizedAdd(b_c, wt_b_c);
		let s_b_o = quantizedAdd(b_o, wt_b_o);
		
		s[63:56] = pack(s_a_i);
		s[55:48] = pack(s_a_f);
		s[47:40] = pack(s_a_c);
		s[39:32] = pack(s_a_o);
		s[31:24] = pack(s_b_i);
		s[23:16] = pack(s_b_f);
		s[15:8] = pack(s_b_c);
		s[7:0] = pack(s_b_o);
		
		bias_calc2_s <= s;
		bias_calc2_c <= bias_calc1_c;
		
		calcStage <= BIAS3;
	endrule
	
	rule calcBias3(calcStage == BIAS3);
		
		Bit#(64) s = bias_calc2_s;
		
		Int#(8) s_a_i = unpack(s[63:56]);
		Int#(8) s_a_f = unpack(s[55:48]);
		Int#(8) s_a_c = unpack(s[47:40]);
		Int#(8) s_a_o = unpack(s[39:32]);
		
		Int#(8) s_b_i = unpack(s[31:24]);
		Int#(8) s_b_f = unpack(s[23:16]);
		Int#(8) s_b_c = unpack(s[15:8]);
		Int#(8) s_b_o = unpack(s[7:0]);
		
		Bit#(64) hs = 0;
		
		let hs_a_i = hardSigmoid(s_a_i);
		let hs_a_f = hardSigmoid(s_a_f);
		let hs_a_c = hardSigmoid(s_a_c);
		let hs_a_o = hardSigmoid(s_a_o);
		
		let hs_b_i = hardSigmoid(s_b_i);
		let hs_b_f = hardSigmoid(s_b_f);
		let hs_b_c = hardSigmoid(s_b_c);
		let hs_b_o = hardSigmoid(s_b_o);
		
		hs[63:56] = pack(hs_a_i);
		hs[55:48] = pack(hs_a_f);
		hs[47:40] = pack(hs_a_c);
		hs[39:32] = pack(hs_a_o);
		hs[31:24] = pack(hs_b_i);
		hs[23:16] = pack(hs_b_f);
		hs[15:8] = pack(hs_b_c);
		hs[7:0] = pack(hs_b_o);
	
		bias_calc3_hs <= hs;
		bias_calc3_c <= bias_calc2_c;
		
		calcStage <= BIAS4;
	endrule
	
	rule calcBias4(calcStage == BIAS4);
		Bit#(64) hs = bias_calc3_hs;
		Bit#(16) c = bias_calc3_c;
		
		Int#(8) hs_a_i = unpack(hs[63:56]);
		Int#(8) hs_a_f = unpack(hs[55:48]);
		Int#(8) hs_a_c = unpack(hs[47:40]);
		
		Int#(8) hs_b_i = unpack(hs[31:24]);
		Int#(8) hs_b_f = unpack(hs[23:16]);
		Int#(8) hs_b_c = unpack(hs[15:8]);
		
		Int#(8) c_a_prev = unpack(c[15:8]);
		Int#(8) c_b_prev = unpack(c[7:0]);
		
		Bit#(16) hs_o = 0;
		hs_o[15:8] = hs[39:32];
		hs_o[7:0] = hs[7:0];
		
		Bit#(16) fprev = 0;
		Bit#(16) ic = 0;
		
		let a_f_prev = quantizedMult(hs_a_f, c_a_prev);
		let b_f_prev = quantizedMult(hs_b_f, c_b_prev);
		
		fprev[15:8] = pack(a_f_prev);
		fprev[7:0] = pack(b_f_prev);
		
		let a_ic = quantizedMult(hs_a_i, hs_a_c);
		let b_ic = quantizedMult(hs_b_i, hs_b_c);
		
		ic[15:8] = pack(a_ic);
		ic[7:0] = pack(b_ic);
		
		bias_calc4_o <= hs_o;
		bias_calc4_ic <= ic;
		bias_calc4_fprev <= fprev;
		
		calcStage <= BIAS5;
	endrule
	
	rule calcBias5(calcStage == BIAS5);
		Bit#(16) fprev = bias_calc4_fprev;
		Int#(8) a_f_prev = unpack(fprev[15:8]);
		Int#(8) b_f_prev = unpack(fprev[7:0]);
		
		Bit#(16) ic = bias_calc4_ic;
		Int#(8) a_ic = unpack(ic[15:8]);
		Int#(8) b_ic = unpack(ic[7:0]);
		
		Bit#(16) ifc = 0;
		
		let a_ifc = quantizedAdd(a_f_prev, a_ic);
		let b_ifc = quantizedAdd(b_f_prev, b_ic);
		
		ifc[15:8] = pack(a_ifc);
		ifc[7:0] = pack(b_ifc);
		
		bias_calc5_o <= bias_calc4_o;
		bias_calc5_ifc <= ifc;
		
		calcStage <= BIAS6;
	endrule
	
	rule calcBias6(calcStage == BIAS6);
		Bit#(16) ifc = bias_calc5_ifc;
		Int#(8) a_ifc = unpack(ifc[15:8]);
		Int#(8) b_ifc = unpack(ifc[7:0]);
		
		let hs_a_ifc = hardSigmoid(a_ifc);
		let hs_b_ifc = hardSigmoid(b_ifc);
		
		Bit#(16) hs_ifc = 0;
		ifc[15:8] = pack(hs_a_ifc);
		ifc[7:0] = pack(hs_b_ifc);
	
		bias_calc6_hs_ifc <= hs_ifc;
		bias_calc6_o <= bias_calc5_o;
		bias_calc6_ifc <= bias_calc5_ifc;
		
		calcStage <= BIAS7;
	endrule
	
	rule calcBias7(calcStage == BIAS7);
		Bit#(16) hs_ifc = bias_calc6_hs_ifc;
		Int#(8) hs_a_ifc = unpack(hs_ifc[15:8]);
		Int#(8) hs_b_ifc = unpack(hs_ifc[7:0]);
		
		Bit#(16) ifc = bias_calc6_ifc;
		Int#(8) a_ifc = unpack(ifc[15:8]);
		Int#(8) b_ifc = unpack(ifc[7:0]);
		
		Bit#(16) hs_o = bias_calc6_o;
		Int#(8) hs_a_o = unpack(hs_o[15:8]);
		Int#(8) hs_b_o = unpack(hs_o[7:0]);
	
		let a_ifco = quantizedMult(hs_a_o, hs_a_ifc);
		let b_ifco = quantizedMult(hs_b_o, hs_b_ifc);
		
		bram_hidden.a.put(True, zeroExtend(lstm_unit_counter), pack(a_ifc));
		bram_hidden.b.put(True, zeroExtend(lstm_unit_counter), pack(b_ifc));
		
		bram_carry.a.put(True, zeroExtend(lstm_unit_counter), pack(a_ifco));
		bram_carry.b.put(True, zeroExtend(lstm_unit_counter), pack(b_ifco));
		
		calcStage <= BIASFETCH;
	endrule
	
	method Action processWeight(Bit#(64) weights);
		spramQ.enq(weights);
		`ifdef BSIM
		$display("processWeight %u", weights);
		`endif
	endmethod
	
	method Action processInput(Int#(8) in);
		inputQ.enq(in);
		`ifdef BSIM
		$display("processInput %d", in);
		`endif
	endmethod
		
	method Action start;
		if (reqStage == INIT) begin
		
			reqStage <= INPUT;
			state <= CALC;
			
			`ifdef BSIM
			$display("lstm start");
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
	
	
endmodule: mkLSTM