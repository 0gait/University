.class public Main
.super java/lang/Object

;
; standard initializer
.method public <init>()V
   aload_0
   invokenonvirtual java/lang/Object/<init>()V
   return
.end method

.method public static main([Ljava/lang/String;)V
       ; set limits used by this method
       .limit locals 10 
       .limit stack 256

       ; setup local variables:

       ;    1 - the PrintStream object held in java.lang.System.out
       getstatic java/lang/System/out Ljava/io/PrintStream;

       ; place your bytecodes between START and END

       ; START ======================================================================

	; initialize frame pointer SL stored in local #4 to null
	aconst_null
	astore 4	

	; SAMPLE code for the following expresion
	; NOTE it relies on two frames defined in f0.j and f1.j
	;
	; def x = 1 
	; in 
	;   def y = 2 + x
	;   in
	; 	x+y
	;   end 
	; + x
	; end


	; def x = 1 in 
	; create frame for this def block (f0)
	new f0
	dup
        invokespecial f0/<init>()V
	dup
	; store SL in new frame
	aload 4
	putfield f0/sl Ljava/lang/Object;
	; update SL
	astore 4

	; store value of x
	aload 4
	sipush 1
	putfield f0/x0 I

	; start def body
	; def y = 2 + x in 
	; create frame for this def block (f1)
	new f1
	dup
       invokespecial f1/<init>()V
	dup
	; store SL in new frame
	aload 4
	putfield f1/sl Lf0;
	astore 4

	; store value of y
	aload 4

	; 2 + x
	sipush 2

       ; fetch x coordinates (1,0)
	aload 4
	getfield f1/sl Lf0;
	getfield f0/x0 I

	iadd
	; now 2 + x is on top of stack

	putfield f1/x0 I

	; start def body
	; x + y

	; fetch x coordinates (1,0)
	aload 4
	getfield f1/sl Lf0;
	getfield f0/x0 I

	; fetch y coordinates (0,0)
	aload 4
	getfield f1/x0 I

	iadd
	; now x + y is on top of stack

	; end scope, retrieve static link from farme and update SL
       aload 4
	getfield f1/sl Lf0;
	astore 4	

	; fetch x coordinates (0,0)
	aload 4
	getfield f0/x0 I

	iadd
	; now def x = 1 .. end + x is on top of stack

	; end scope, retrieve static link from farme and update SL
	aload 4
	getfield f0/sl Ljava/lang/Object;
	astore 4

       ; END ====================================================================

       ; convert to String;
       invokestatic java/lang/String/valueOf(I)Ljava/lang/String;
       ; call println 
       invokevirtual java/io/PrintStream/println(Ljava/lang/String;)V

       return

.end method
