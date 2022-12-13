.class public closure_1
.super java/lang/Object
.implements closure_interface_int_int
.field public sl Lframe_2;
.method public <init>()V
	aload_0
	invokenonvirtual java/lang/Object/<init>()V
	return
.end method
.method public call(I)I
	.limit locals 4
	.limit stack 256
	new frame_3
	dup
	invokespecial frame_3/<init>()V
	dup
	aload_0
	getfield closure_1/sl Lframe_2;
	putfield frame_3/sl Lframe_2;
	dup
	iload 1
	putfield frame_3/v0 I
	astore_3
	aload_3
	getfield frame_3/v0 I
	sipush 2
	imul
	ireturn
.end method
