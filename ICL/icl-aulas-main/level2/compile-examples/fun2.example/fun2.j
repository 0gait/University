.class public fun2
.super java/lang/Object
.method public <init>()V
	aload_0
	invokenonvirtual java/lang/Object/<init>()V
	return
.end method
.method public static main([Ljava/lang/String;)V
	; set limits used by this method
	.limit locals 10
	.limit stack 256
	; initialize SL variable to null
	aconst_null
	astore_3
	; ---- END OF INTRO CODE
	new frame_0
	dup
	invokespecial frame_0/<init>()V
	dup
	aload_3
	putfield frame_0/sl Ljava/lang/Object;
	dup
	astore_3
	dup
	new closure_0
	dup
	invokespecial closure_0/<init>()V
	dup
	aload_3
	putfield closure_0/sl Lframe_0;
	putfield frame_0/v0 Ljava/lang/Object;
	pop
	new frame_2
	dup
	invokespecial frame_2/<init>()V
	dup
	aload_3
	putfield frame_2/sl Lframe_0;
	dup
	astore_3
	dup
	aload_3
	getfield frame_2/sl Lframe_0;
	getfield frame_0/v0 Ljava/lang/Object;
	checkcast closure_interface_closure_int
	new closure_1
	dup
	invokespecial closure_1/<init>()V
	dup
	aload_3
	putfield closure_1/sl Lframe_2;
	invokeinterface closure_interface_closure_int/call(Ljava/lang/Object;)I 2
	putfield frame_2/v0 I
	pop
	aload_3
	getfield frame_2/v0 I
	sipush 2
	iadd
	dup
	getstatic java/lang/System/out Ljava/io/PrintStream;
	swap
	invokevirtual java/io/PrintStream/println(I)V
	aload_3
	getfield frame_2/sl Lframe_0;
	astore_3
	aload_3
	getfield frame_0/sl Ljava/lang/Object;
	astore_3
	; ---- START OF EPILOGUE CODE
	return
.end method
