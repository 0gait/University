.class public closure_0
.super java/lang/Object
.implements closure_interface_int_int
.field public sl Lframe_0;
.method public <init>()V
	aload_0
	invokenonvirtual java/lang/Object/<init>()V
	return
.end method
.method public call(I)I
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
	iload 1
	putfield frame_1/v0 I
	astore_3
	aload_3
	getfield frame_1/v0 I
	sipush 0
	if_icmpeq L1
	iconst_0
	goto L2
L1:
	iconst_1
L2:
	ifeq L3
	sipush 1
	goto L4
L3:
	aload_3
	getfield frame_1/v0 I
	aload_3
	getfield frame_1/sl Lframe_0;
	getfield frame_0/v0 Ljava/lang/Object;
	checkcast closure_interface_int_int
	aload_3
	getfield frame_1/v0 I
	sipush 1
	isub
	invokeinterface closure_interface_int_int/call(I)I 2
	imul
L4:
	ireturn
.end method
