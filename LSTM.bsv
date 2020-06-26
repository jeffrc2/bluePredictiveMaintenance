import BRAMCore::*;


typedef enum {A, B} Port deriving(Bits,Eq);

interface LSTMIfc#(numeric type kernellen, numeric type recurrentlen, numeric type biaslen);
	method ActionValue#(Bit#(14)) deqBRAMReq;
	method Action process(Bit#(8) data);
	
endinterface

module mkLSTM(LSTMIfc#(kernellen, recurrentlen, biaslen));

	Reg#(Bit#(9)) input_counter <- mkReg(0);
	BRAM_DUAL_PORT#(Bit#(8), Bit#(16)) bram_x_if <- mkBRAMCore2(200, False);
	BRAM_DUAL_PORT#(Bit#(8), Bit#(16)) bram_x_co <- mkBRAMCore2(200, False);
	BRAM_DUAL_PORT#(Bit#(8), Bit#(16)) bram_y_if <- mkBRAMCore2(200, False);
	BRAM_DUAL_PORT#(Bit#(8), Bit#(16)) bram_y_co <- mkBRAMCore2(200, False);
	BRAM_DUAL_PORT#(Bit#(8), Bit#(16)) bram_hidden <- mkBRAMCore2(200, False);
	
	BRAM_DUAL_PORT#(Bit#(9), Bit#(8)) bram_input1 <- mkBRAMCore2(512, False);
	BRAM_DUAL_PORT#(Bit#(9), Bit#(8)) bram_input2 <- mkBRAMCore2(512, False);
	BRAM_DUAL_PORT#(Bit#(9), Bit#(8)) bram_input3 <- mkBRAMCore2(226, False);
	
	method Action hiddenReadIn(Bit#(64) data);
		Int#(32) a = data[31:0];
		Int#(32) b = data[63:32];
		hiddenCalc(a, A);
		hiddenCalc(b, B);
	endmethod
	
	
	method Action hiddenCalc(Bit#(32) data, Port side);
		Int#(8) rt_i = unpack(data[7:0]));
		Int#(8) rt_f = unpack(data[:8]);
		Int#(8) rt_c = unpack(data[23:16]);
		Int#(8) rt_o = unpack(data[31:24]);
		if (counter == 25) hiddenQ.deq;
		Int#(8) hiddenIn = unpack(hiddenQ.first);

		Int#(16) y_if;
		Int#(16) y_co;
		case (side) matches
			A: begin
				let y_if = bram_y_if.a.read;
				let y_co = bram_y_co.a.read;
			end
			B: begin
				let y_if = bram_y_if.b.read;
				let y_co = bram_y_co.b.read;
			end
		endcase
		
		Int#(8) y_i = y_if[7:0];
		Int#(8) y_f = y_if[15:8]; 
		Int#(8) y_c = y_co[7:0];
		Int#(8) y_o = y_co[15:8]; 
		
		let p_i = quantizedMult(dataIn, rt_i);
		let p_f = quantizedMult(dataIn, rt_f);
		let p_c = quantizedMult(dataIn, rt_c);
		let p_o = quantizedMult(dataIn, rt_o);		
		let s_i = quantizedAdd(y_i, p_i);
		let s_f = quantizedAdd(y_f, p_f);
		let s_c = quantizedAdd(y_c, p_c);
		let s_o = quantizedAdd(y_o, p_o);	
		
		case (side) matches
			A: begin
				bram_x_if.a.put(True, currAddr, {s_f, s_i});
				bram_x_co.a.put(True, currAddr, {s_f, s_i});
			end
			B: begin
				bram_x_if.b.put(True, currAddr, {s_f, s_i});
				bram_x_co.b.put(True, currAddr, {s_f, s_i});
			end
				
		end
	endmethod
	
	method Action inputReadIn(Bit#(64) data);
		Int#(32) a = data[31:0];
		Int#(32) b = data[63:32];
		inputCalc(a, A);
		inputCalc(b, B);
	endmethod
	
	function Action inputCalc(Bit#(32) data, Port side);
		
		Int#(8) wt_i = unpack(data[7:0]));
		Int#(8) wt_f = unpack(data[15:8]);
		Int#(8) wt_c = unpack(data[23:16]);
		Int#(8) wt_o = unpack(data[31:24]);
		
		if (counter == 25) inputQ.deq;
		Int#(8) dataIn = unpack(inputQ.first);
		

		Int#(16) x_if;
		Int#(16) x_co;
		case (side) matches
			A: begin
				let x_if = bram_x_if.a.read;
				let x_co = bram_x_co.a.read;
			end
			B: begin
				let x_if = bram_x_if.b.read;
				let x_co = bram_x_co.b.read;
			end
		endcase
		
		Int#(8) x_i = x_if[7:0];
		Int#(8) x_f = x_if[15:8]; 
		Int#(8) x_c = x_co[7:0];
		Int#(8) x_o = x_co[15:8]; 
		
		let p_i = quantizedMult(dataIn, wt_i);
		let p_f = quantizedMult(dataIn, wt_f);
		let p_c = quantizedMult(dataIn, wt_c);
		let p_o = quantizedMult(dataIn, wt_o);		
		let s_i = quantizedAdd(x_i, p_i);
		let s_f = quantizedAdd(x_f, p_f);
		let s_c = quantizedAdd(x_c, p_c);
		let s_o = quantizedAdd(x_o, p_o);	
		
		case (side) matches
			A: begin
				bram_x_if.a.put(True, currAddr, {s_f, s_i});
				bram_x_co.a.put(True, currAddr, {s_f, s_i});
			end
			B: begin
				bram_x_if.b.put(True, currAddr, {s_f, s_i});
				bram_x_co.b.put(True, currAddr, {s_f, s_i});
			end
				
		end
		
	endfunction
	
	method Action process(Bit#(8) data);
		if (counter % 2 == 0) bram_x.a.put(True, counter, data);
		else bram_x.a.put(False, counter, ?);
		counter <= counter + 1;
	endmethod
	
	function Int#(8) quantizedAdd(Int#(8) x, Int#(8) y);
		Int#(16) sx = signExtend(x) - zeropoint;
		Int#(16) sy = signExtend(y) - zeropoint;
		let dx = sx / 2;
		let dy = sy / 2;
		let s = dx + dy;
		let s2 = s*2;
		return truncate(s2 + zeropoint);
	endfunction

	function Int#(8) quantizedMult(Int#(8) x, Int#(8) y);
		Int#(16) sx = signExtend(x);
		Int#(16) sy = signExtend(y);
		let 
		let p = sx * sy;
		let s = p / invscale;
		return truncate(s + zeropoint);
	endfunction

	function Int#(8) hardSigmoid(Int#(8) d);
		Bit#(8) lowerlimit = 8'b00000000;
		Bit#(8) upperlimit = 8'b00000000;
		if (d <= unpack(lowerlimit)) return zeropoint;
		else if (d >= unpack(upperlimit)) return onepoint;
		else return quantizedAdd(quantizedMult(alpha, d), offset);
	endfunction
	
	
	
endmodule: mkLSTM