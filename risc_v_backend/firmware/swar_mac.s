	.file	"swar_mac.c"
	.option nopic
	.attribute arch, "rv32i2p1_m2p0_zmmul1p0"
	.attribute unaligned_access, 0
	.attribute stack_align, 16
	.text
	.align	2
	.globl	swar_mac_8bit
	.type	swar_mac_8bit, @function
swar_mac_8bit:
	ble	a2,zero,.L4
	slli	a2,a2,1
	mv	a5,a0
	add	a2,a0,a2
	li	a0,0
.L3:
	lb	a4,0(a5)
	lb	a3,0(a1)
	addi	a5,a5,2
	addi	a1,a1,1
	mul	a4,a4,a3
	add	a0,a0,a4
	bne	a2,a5,.L3
	ret
.L4:
	li	a0,0
	ret
	.size	swar_mac_8bit, .-swar_mac_8bit
	.align	2
	.globl	swar_mac_4bit_d4
	.type	swar_mac_4bit_d4, @function
swar_mac_4bit_d4:
	ble	a2,zero,.L10
	addi	sp,sp,-16
	slli	a2,a2,2
	li	a6,-1840701440
	mv	a7,a0
	sw	s0,12(sp)
	sw	s1,8(sp)
	sw	s2,4(sp)
	add	t1,a1,a2
	addi	a6,a6,1171
	li	a0,0
.L9:
	lhu	a5,0(a7)
	lb	t2,0(a1)
	lb	t5,1(a1)
	slli	a3,a5,28
	srai	a4,a3,28
	slli	t6,a4,7
	slli	a3,a5,24
	sub	t6,t6,a4
	srai	t3,a3,28
	slli	a3,a5,20
	srai	a4,a3,28
	mulh	a3,t6,a6
	slli	a2,t3,7
	sub	a2,a2,t3
	slli	t0,a4,7
	sub	t0,t0,a4
	slli	t3,a5,16
	srai	a5,t3,28
	slli	t3,a5,7
	sub	t3,t3,a5
	srai	a5,t6,31
	mulh	a4,a2,a6
	add	a3,a3,t6
	srai	a3,a3,2
	sub	a3,a3,a5
	srai	s2,a2,31
	lb	t4,2(a1)
	lb	t6,3(a1)
	srai	s1,t0,31
	srai	s0,t3,31
	addi	a1,a1,4
	mulh	a5,t0,a6
	add	a4,a4,a2
	srai	a4,a4,2
	sub	a4,a4,s2
	addi	a7,a7,2
	mulh	a2,t3,a6
	add	a5,a5,t0
	srai	a5,a5,2
	sub	a5,a5,s1
	mul	a3,a3,t2
	add	a2,a2,t3
	srai	a2,a2,2
	sub	a2,a2,s0
	mul	a4,a4,t5
	add	a3,a3,a0
	mul	a5,a5,t4
	add	a4,a4,a3
	mul	a2,a2,t6
	add	a5,a5,a4
	add	a0,a2,a5
	bne	t1,a1,.L9
	lw	s0,12(sp)
	lw	s1,8(sp)
	lw	s2,4(sp)
	addi	sp,sp,16
	jr	ra
.L10:
	li	a0,0
	ret
	.size	swar_mac_4bit_d4, .-swar_mac_4bit_d4
	.align	2
	.globl	swar_mac_general
	.type	swar_mac_general, @function
swar_mac_general:
	addi	sp,sp,-112
	sw	ra,108(sp)
	sw	s11,60(sp)
	ble	a6,zero,.L25
	slli	a6,a6,1
	sw	s5,84(sp)
	mv	s5,a0
	add	a0,a0,a6
	sw	a0,28(sp)
	ble	a5,zero,.L26
	sw	a2,36(sp)
	li	a2,-1840701440
	addi	a2,a2,1171
	sw	a2,24(sp)
	li	a2,1431654400
	sw	s0,104(sp)
	sw	a5,20(sp)
	addi	a2,a2,1366
	add	s0,a3,a5
	slli	a5,a5,1
	sw	s2,96(sp)
	sw	s3,92(sp)
	sw	s4,88(sp)
	sw	s6,80(sp)
	sw	s1,100(sp)
	sw	s7,76(sp)
	sw	s8,72(sp)
	sw	s9,68(sp)
	sw	s10,64(sp)
	sw	a3,40(sp)
	mv	s2,a1
	sw	a4,16(sp)
	sw	a2,44(sp)
	sw	a5,32(sp)
	sw	zero,12(sp)
	li	s11,0
	li	s4,32
	li	s3,8
	li	s6,4
.L24:
	lw	a5,40(sp)
	lw	a4,12(sp)
	lhu	s1,0(s5)
	lw	s10,16(sp)
	add	s8,a5,a4
	lw	a5,36(sp)
	add	s9,a5,a4
	j	.L23
.L20:
	li	a4,2
	beq	a0,a4,.L36
	li	a4,3
	beq	a0,a4,.L37
	addi	a0,a0,-1
	li	a4,1
	sll	a0,a4,a0
	addi	a0,a0,-1
	beq	a0,zero,.L19
	slli	a3,a5,7
	sub	a5,a3,a5
	div	a5,a5,a0
.L19:
	lhu	a3,0(s10)
	add	a3,s2,a3
	lb	a3,0(a3)
	mul	a5,a3,a5
	add	s11,s11,a5
.L18:
	addi	s8,s8,1
	addi	s10,s10,2
	addi	s9,s9,1
	beq	s8,s0,.L38
.L23:
	lbu	s7,0(s8)
	beq	s7,zero,.L18
	mv	a0,s7
	call	__popcountsi2
	lbu	a5,0(s9)
	sub	a7,s4,a0
	srl	a5,s1,a5
	and	a5,a5,s7
	sll	a5,a5,a7
	sra	a5,a5,a7
	beq	a0,s3,.L19
	bne	a0,s6,.L20
	slli	a3,a5,7
	sub	a3,a3,a5
	lw	a5,24(sp)
	srai	a0,a3,31
	mulh	a5,a3,a5
	add	a5,a5,a3
	srai	a5,a5,2
	sub	a5,a5,a0
	j	.L19
.L38:
	lw	a5,20(sp)
	lw	a4,32(sp)
	addi	s5,s5,2
	add	s0,s0,a5
	lw	a5,16(sp)
	add	a5,a5,a4
	sw	a5,16(sp)
	lw	a4,20(sp)
	lw	a5,12(sp)
	add	a5,a5,a4
	sw	a5,12(sp)
	lw	a5,28(sp)
	bne	s5,a5,.L24
	lw	s0,104(sp)
	lw	s1,100(sp)
	lw	s2,96(sp)
	lw	s3,92(sp)
	lw	s4,88(sp)
	lw	s5,84(sp)
	lw	s6,80(sp)
	lw	s7,76(sp)
	lw	s8,72(sp)
	lw	s9,68(sp)
	lw	s10,64(sp)
.L16:
	lw	ra,108(sp)
	mv	a0,s11
	lw	s11,60(sp)
	addi	sp,sp,112
	jr	ra
.L36:
	slli	a3,a5,7
	sub	a5,a3,a5
	j	.L19
.L37:
	slli	a3,a5,7
	sub	a3,a3,a5
	lw	a5,44(sp)
	mulh	a5,a3,a5
	srai	a3,a3,31
	sub	a5,a5,a3
	j	.L19
.L25:
	lw	ra,108(sp)
	li	s11,0
	mv	a0,s11
	lw	s11,60(sp)
	addi	sp,sp,112
	jr	ra
.L26:
	lw	s5,84(sp)
	li	s11,0
	j	.L16
	.size	swar_mac_general, .-swar_mac_general
	.globl	__popcountsi2
	.ident	"GCC: (xPack GNU RISC-V Embedded GCC x86_64) 15.2.0"
	.section	.note.GNU-stack,"",@progbits
