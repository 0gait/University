.class public closure_0
.super java/lang/Object
.implements closure_interface_closure_int
.field public sl Lframe_0;
.method public <init>()V
	aload_0
	invokenonvirtual java/lang/Object/<init>()V
	return
.end method
.method public call(Ljava/lang/Object;)I
	.limit locals 4
	.limit stack 256
	new frame_1
	dup
	invokespecial frame_1/<init>()V
	dup
	aload_0
	getfield closure_0/sl Lframe_0;
	putfield frame_1/sl Lframe_0;
	dup
	aload 1
	putfield frame_1/v0 Ljava/lang/Object;
	astore_3
	aload_3
	getfield frame_1/v0 Ljava/lang/Object;
	checkcast closure_interface_int_int
	sipush 10
	invokeinterface closure_interface_int_int/call(I)I 2
	ireturn
.end method
