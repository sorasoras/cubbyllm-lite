
B:\git\cubbyllm-lite\kernels\_ab_clang_fwg_wpe.hsaco:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001b00 <moe_v19>:
	s_load_b96 s[20:22], s[0:1], 0x38                          // 000000001B00: F400A500 F8000038
	s_wait_kmcnt 0x0                                           // 000000001B08: BFC70000
	s_cmp_ge_i32 ttmp9, s22                                    // 000000001B0C: BF031675
	s_cbranch_scc1 1209                                        // 000000001B10: BFA204B9 <moe_v19+0x12f8>
	v_and_b32_e32 v1, 31, v0                                   // 000000001B14: 3602009F
	v_bfe_u32 v7, v0, 10, 10                                   // 000000001B18: D6100007 02291500
	s_load_b256 s[12:19], s[0:1], 0x20                         // 000000001B20: F4006300 F8000020
	v_lshrrev_b32_e32 v2, 6, v0                                // 000000001B28: 32040086
	v_and_b32_e32 v77, 15, v0                                  // 000000001B2C: 369A008F
	s_load_b256 s[4:11], s[0:1], 0x0                           // 000000001B30: F4006100 F8000000
	v_lshl_or_b32 v5, v7, 5, v1                                // 000000001B38: D6560005 04050B07
	v_lshrrev_b32_e32 v1, 10, v0                               // 000000001B40: 3202008A
	v_and_b32_e32 v6, 0xf0, v2                                 // 000000001B44: 360C04FF 000000F0
	v_and_b32_e32 v8, 0x3ff, v0                                // 000000001B4C: 361000FF 000003FF
	v_bfe_u32 v12, v0, 4, 1                                    // 000000001B54: D610000C 02050900
	v_lshrrev_b32_e32 v9, 1, v5                                // 000000001B5C: 32120A81
	v_and_b32_e32 v64, 0x7f, v5                                // 000000001B60: 36800AFF 0000007F
	v_lshrrev_b16 v5.l, 1, v5.l                                // 000000001B68: D7390005 02020A81
	v_mov_b16_e32 v5.h, 0                                      // 000000001B70: 7F0A3880
	v_bfe_u32 v10, v1, 2, 8                                    // 000000001B74: D610000A 02210501
	v_or_b32_e32 v11, v6, v77                                  // 000000001B7C: 38169B06
	v_lshlrev_b32_e32 v84, 3, v12                              // 000000001B80: 30A81883
	v_lshlrev_b32_e32 v85, 10, v12                             // 000000001B84: 30AA188A
	v_mul_lo_u32 v5, s21, v5                                   // 000000001B88: D72C0005 02020A15
	v_lshlrev_b32_e32 v78, 1, v10                              // 000000001B90: 309C1481
	v_lshlrev_b32_e32 v82, 10, v10                             // 000000001B94: 30A4148A
	v_lshrrev_b32_e32 v10, 1, v0                               // 000000001B98: 32140081
	v_and_b32_e32 v0, 1, v0                                    // 000000001B9C: 36000081
	s_load_b32 s36, s[0:1], 0x48                               // 000000001BA0: F4000900 F8000048
	s_wait_kmcnt 0x0                                           // 000000001BA8: BFC70000
	s_load_b32 s40, s[14:15], 0x0                              // 000000001BAC: F4000A07 F8000000
	v_mul_lo_u32 v1, s21, v9                                   // 000000001BB4: D72C0001 02021215
	v_and_or_b32 v87, v10, 8, v6                               // 000000001BBC: D6570057 0419110A
	v_ashrrev_i32_e32 v6, 31, v5                               // 000000001BC4: 340C0A9F
	v_lshlrev_b32_e32 v12, 3, v0                               // 000000001BC8: 30180083
	v_lshlrev_b16 v0.l, 5, v7.l                                // 000000001BCC: D7380000 02020E85
	v_and_b16 v0.h, v8.l, 31 op_sel:[0,0,1]                    // 000000001BD4: D7624000 02013F08
	v_mad_co_u64_u32 v[3:4], null, s20, v78, v[64:65]          // 000000001BDC: D6FE7C03 05029C14
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]                        // 000000001BE4: 3E0A0A82
	v_lshlrev_b32_e32 v2, 1, v8                                // 000000001BE8: 30041081
	s_cmp_gt_i32 s21, 0                                        // 000000001BEC: BF028015
	v_or_b16 v0.l, v0.l, v0.h op_sel:[0,1,0]                   // 000000001BF0: D7631000 02020100
	v_lshlrev_b32_e32 v79, 3, v64                              // 000000001BF8: 309E8083
	v_mul_u32_u24_e32 v83, 24, v11                             // 000000001BFC: 16A61698
	v_add_co_u32 v5, vcc_lo, v12, v5                           // 000000001C00: D7006A05 02020B0C
	v_lshlrev_b32_e32 v86, 3, v77                              // 000000001C08: 30AC9A83
	v_mad_u32_u24 v10, v11, 24, 0                              // 000000001C0C: D60B000A 0201310B
	v_add_nc_u32_e32 v11, 0, v85                               // 000000001C14: 4A16AA80
	v_add_co_ci_u32_e64 v6, null, 0, v6, vcc_lo                // 000000001C18: D5207C06 01AA0C80
	v_cmp_gt_u32_e64 s2, 16, v7                                // 000000001C20: D44C0002 02020E90
	v_cmp_gt_u32_e64 s3, 8, v7                                 // 000000001C28: D44C0003 02020E88
	s_cselect_b32 s33, -1, 0                                   // 000000001C30: 982180C1
	s_add_co_i32 s18, s21, -1                                  // 000000001C34: 8112C115
	v_add_nc_u32_e32 v7, 4, v78                                // 000000001C38: 4A0E9C84
	v_and_b32_e32 v0, 0x7f, v0                                 // 000000001C3C: 360000FF 0000007F
	v_add_nc_u32_e32 v8, 8, v78                                // 000000001C44: 4A109C88
	s_lshr_b32 s19, s18, 2                                     // 000000001C48: 85138212
	v_add_co_u32 v5, vcc_lo, s4, v5                            // 000000001C4C: D7006A05 02020A04
	v_and_b32_e32 v13, 2, v2                                   // 000000001C54: 361A0482
	v_ashrrev_i32_e32 v2, 31, v1                               // 000000001C58: 3404029F
	v_ashrrev_i32_e32 v4, 31, v3                               // 000000001C5C: 3408069F
	v_mad_u32_u24 v14, v9, 24, 0                               // 000000001C60: D60B000E 02013109
	v_mul_u32_u24_e32 v80, 24, v9                              // 000000001C68: 16A01298
	v_add_nc_u32_e32 v9, 0, v79                                // 000000001C6C: 4A129E80
	s_add_co_i32 s25, s19, 1                                   // 000000001C70: 81198113
	s_wait_alu depctr_va_vcc(0)                                // 000000001C74: BF88FF9D
	v_add_co_ci_u32_e64 v6, null, s5, v6, vcc_lo               // 000000001C78: D5207C06 01AA0C05
	v_add_nc_u32_e32 v91, v11, v86                             // 000000001C80: 4AB6AD0B
	s_cmp_lg_u32 s19, 0                                        // 000000001C84: BF078013
	v_mad_co_u64_u32 v[65:66], null, s20, v7, v[0:1]           // 000000001C88: D6FE7C41 04020E14
	v_mad_co_u64_u32 v[66:67], null, s20, v8, v[0:1]           // 000000001C90: D6FE7C42 04021014
	s_cselect_b32 s37, -1, 0                                   // 000000001C98: 982580C1
	s_and_b32 s38, s25, 0x7ffffffe                             // 000000001C9C: 8B26FF19 7FFFFFFE
	v_add_co_u32 v88, vcc_lo, v5, 16                           // 000000001CA4: D7006A58 02012105
	s_bitcmp0_b32 s18, 2                                       // 000000001CAC: BF0C8212
	v_lshlrev_b64_e32 v[67:68], 2, v[1:2]                      // 000000001CB0: 3E860282
	v_lshlrev_b64_e32 v[69:70], 2, v[3:4]                      // 000000001CB4: 3E8A0682
	v_lshl_add_u32 v81, v13, 2, v14                            // 000000001CB8: D6460051 0439050D
	s_wait_alu depctr_va_vcc(0)                                // 000000001CC0: BF88FF9D
	v_add_co_ci_u32_e64 v89, null, 0, v6, vcc_lo               // 000000001CC4: D5207C59 01AA0C80
	v_add_nc_u32_e32 v90, v9, v82                              // 000000001CCC: 4AB4A509
	v_add_nc_u32_e32 v92, v10, v84                             // 000000001CD0: 4AB8A90A
	v_lshlrev_b32_e32 v93, 2, v13                              // 000000001CD4: 30BA1A82
	v_add_nc_u32_e32 v94, 0x1800, v91                          // 000000001CD8: 4ABCB6FF 00001800
	s_cselect_b32 s39, -1, 0                                   // 000000001CE0: 982780C1
	s_ashr_i32 s1, s20, 31                                     // 000000001CE4: 86019F14
	s_mov_b32 s0, s20                                          // 000000001CE8: BE800014
	s_mov_b32 s24, ttmp9                                       // 000000001CEC: BE980075
	s_mul_i32 s23, s21, s20                                    // 000000001CF0: 96171415
	s_mov_b32 s15, 0                                           // 000000001CF4: BE8F0080
	s_lshl_b64 s[18:19], s[0:1], 2                             // 000000001CF8: 84928200
	s_lshl_b32 s41, s20, 3                                     // 000000001CFC: 84298314
	s_branch 549                                               // 000000001D00: BFA00225 <moe_v19+0xa98>
	v_mov_b32_e32 v56, 0                                       // 000000001D04: 7E700280
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001D08: BF870001
	v_dual_mov_b32 v57, v56 :: v_dual_mov_b32 v58, v56         // 000000001D0C: CA100138 393A0138
	v_dual_mov_b32 v59, v56 :: v_dual_mov_b32 v60, v56         // 000000001D14: CA100138 3B3C0138
	v_dual_mov_b32 v61, v56 :: v_dual_mov_b32 v62, v56         // 000000001D1C: CA100138 3D3E0138
	v_dual_mov_b32 v63, v56 :: v_dual_mov_b32 v48, v56         // 000000001D24: CA100138 3F300138
	v_dual_mov_b32 v49, v56 :: v_dual_mov_b32 v50, v56         // 000000001D2C: CA100138 31320138
	v_dual_mov_b32 v51, v56 :: v_dual_mov_b32 v52, v56         // 000000001D34: CA100138 33340138
	v_dual_mov_b32 v53, v56 :: v_dual_mov_b32 v54, v56         // 000000001D3C: CA100138 35360138
	v_dual_mov_b32 v55, v56 :: v_dual_mov_b32 v40, v56         // 000000001D44: CA100138 37280138
	v_dual_mov_b32 v41, v56 :: v_dual_mov_b32 v42, v56         // 000000001D4C: CA100138 292A0138
	v_dual_mov_b32 v43, v56 :: v_dual_mov_b32 v44, v56         // 000000001D54: CA100138 2B2C0138
	v_dual_mov_b32 v45, v56 :: v_dual_mov_b32 v46, v56         // 000000001D5C: CA100138 2D2E0138
	v_dual_mov_b32 v47, v56 :: v_dual_mov_b32 v32, v56         // 000000001D64: CA100138 2F200138
	v_dual_mov_b32 v33, v56 :: v_dual_mov_b32 v34, v56         // 000000001D6C: CA100138 21220138
	v_dual_mov_b32 v35, v56 :: v_dual_mov_b32 v36, v56         // 000000001D74: CA100138 23240138
	v_dual_mov_b32 v37, v56 :: v_dual_mov_b32 v38, v56         // 000000001D7C: CA100138 25260138
	v_dual_mov_b32 v39, v56 :: v_dual_mov_b32 v24, v56         // 000000001D84: CA100138 27180138
	v_dual_mov_b32 v25, v56 :: v_dual_mov_b32 v26, v56         // 000000001D8C: CA100138 191A0138
	v_dual_mov_b32 v27, v56 :: v_dual_mov_b32 v28, v56         // 000000001D94: CA100138 1B1C0138
	v_dual_mov_b32 v29, v56 :: v_dual_mov_b32 v30, v56         // 000000001D9C: CA100138 1D1E0138
	v_dual_mov_b32 v31, v56 :: v_dual_mov_b32 v16, v56         // 000000001DA4: CA100138 1F100138
	v_dual_mov_b32 v17, v56 :: v_dual_mov_b32 v18, v56         // 000000001DAC: CA100138 11120138
	v_dual_mov_b32 v19, v56 :: v_dual_mov_b32 v20, v56         // 000000001DB4: CA100138 13140138
	v_dual_mov_b32 v21, v56 :: v_dual_mov_b32 v22, v56         // 000000001DBC: CA100138 15160138
	v_dual_mov_b32 v23, v56 :: v_dual_mov_b32 v8, v56          // 000000001DC4: CA100138 17080138
	v_dual_mov_b32 v9, v56 :: v_dual_mov_b32 v10, v56          // 000000001DCC: CA100138 090A0138
	v_dual_mov_b32 v11, v56 :: v_dual_mov_b32 v12, v56         // 000000001DD4: CA100138 0B0C0138
	v_dual_mov_b32 v13, v56 :: v_dual_mov_b32 v14, v56         // 000000001DDC: CA100138 0D0E0138
	v_dual_mov_b32 v15, v56 :: v_dual_mov_b32 v0, v56          // 000000001DE4: CA100138 0F000138
	v_dual_mov_b32 v1, v56 :: v_dual_mov_b32 v2, v56           // 000000001DEC: CA100138 01020138
	v_dual_mov_b32 v3, v56 :: v_dual_mov_b32 v4, v56           // 000000001DF4: CA100138 03040138
	v_dual_mov_b32 v5, v56 :: v_dual_mov_b32 v6, v56           // 000000001DFC: CA100138 05060138
	v_mov_b32_e32 v7, v56                                      // 000000001E04: 7E0E0338
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001E08: BF870001
	v_cvt_f32_i32_e32 v56, v56                                 // 000000001E0C: 7E700B38
	v_add_nc_u32_e32 v71, s25, v87                             // 000000001E10: 4A8EAE19
	v_cvt_f32_i32_e32 v57, v57                                 // 000000001E14: 7E720B39
	v_cvt_f32_i32_e32 v58, v58                                 // 000000001E18: 7E740B3A
	v_cvt_f32_i32_e32 v59, v59                                 // 000000001E1C: 7E760B3B
	v_mul_f32_e32 v56, s40, v56                                // 000000001E20: 10707028
	v_mul_lo_u32 v71, v71, s20                                 // 000000001E24: D72C0047 02002947
	v_mul_f32_e32 v72, s40, v57                                // 000000001E2C: 10907228
	v_cvt_f32_i32_e32 v60, v60                                 // 000000001E30: 7E780B3C
	v_cvt_f32_i32_e32 v61, v61                                 // 000000001E34: 7E7A0B3D
	v_cvt_i32_f32_e32 v76, v56                                 // 000000001E38: 7E981138
	v_mul_f32_e32 v73, s40, v59                                // 000000001E3C: 10927628
	v_cvt_f32_i32_e32 v48, v48                                 // 000000001E40: 7E600B30
	v_mul_f32_e32 v74, s40, v60                                // 000000001E44: 10947828
	v_add3_u32 v57, s26, v77, v71                              // 000000001E48: D6550039 051E9A1A
	v_mul_f32_e32 v71, s40, v58                                // 000000001E50: 108E7428
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)// 000000001E54: BF870224
	v_dual_mul_f32 v75, s40, v61 :: v_dual_mul_f32 v48, s40, v48// 000000001E58: C8C67A28 4B306028
	v_cvt_i32_f32_e32 v95, v72                                 // 000000001E60: 7EBE1148
	v_ashrrev_i32_e32 v58, 31, v57                             // 000000001E64: 3474729F
	v_add_co_u32 v56, vcc_lo, s16, v57                         // 000000001E68: D7006A38 02027210
	v_cvt_i32_f32_e32 v96, v71                                 // 000000001E70: 7EC01147
	v_cvt_i32_f32_e32 v73, v73                                 // 000000001E74: 7E921149
	s_wait_alu depctr_va_vcc(0)                                // 000000001E78: BF88FF9D
	v_add_co_ci_u32_e64 v57, null, s17, v58, vcc_lo            // 000000001E7C: D5207C39 01AA7411
	v_add_co_u32 v58, vcc_lo, v56, s0                          // 000000001E84: D7006A3A 02000138
	v_cvt_f32_i32_e32 v62, v62                                 // 000000001E8C: 7E7C0B3E
	s_wait_alu depctr_va_vcc(0)                                // 000000001E90: BF88FF9D
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 000000001E94: BF870193
	v_add_co_ci_u32_e64 v59, null, s1, v57, vcc_lo             // 000000001E98: D5207C3B 01AA7201
	v_add_co_u32 v60, vcc_lo, v58, s0                          // 000000001EA0: D7006A3C 0200013A
	v_cvt_i32_f32_e32 v97, v74                                 // 000000001EA8: 7EC2114A
	s_wait_alu depctr_va_vcc(0)                                // 000000001EAC: BF88FF9D
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 000000001EB0: BF870193
	v_add_co_ci_u32_e64 v61, null, s1, v59, vcc_lo             // 000000001EB4: D5207C3D 01AA7601
	v_add_co_u32 v71, vcc_lo, v60, s0                          // 000000001EBC: D7006A47 0200013C
	v_cvt_i32_f32_e32 v98, v75                                 // 000000001EC4: 7EC4114B
	s_wait_alu depctr_va_vcc(0)                                // 000000001EC8: BF88FF9D
	s_delay_alu instid0(VALU_DEP_3)                            // 000000001ECC: BF870003
	v_add_co_ci_u32_e64 v72, null, s1, v61, vcc_lo             // 000000001ED0: D5207C48 01AA7A01
	s_clause 0x3                                               // 000000001ED8: BF850003
	global_store_b8 v[56:57], v76, off                         // 000000001EDC: EE06007C 26000000 00000038
	global_store_b8 v[58:59], v95, off                         // 000000001EE8: EE06007C 2F800000 0000003A
	global_store_b8 v[60:61], v96, off                         // 000000001EF4: EE06007C 30000000 0000003C
	global_store_b8 v[71:72], v73, off                         // 000000001F00: EE06007C 24800000 00000047
	v_add_co_u32 v73, vcc_lo, v71, s0                          // 000000001F0C: D7006A49 02000147
	s_wait_alu depctr_va_vcc(0)                                // 000000001F14: BF88FF9D
	v_add_co_ci_u32_e64 v74, null, s1, v72, vcc_lo             // 000000001F18: D5207C4A 01AA9001
	v_mul_f32_e32 v75, s40, v62                                // 000000001F20: 10967C28
	v_cvt_f32_i32_e32 v76, v63                                 // 000000001F24: 7E980B3F
	v_add_co_u32 v62, vcc_lo, v73, s0                          // 000000001F28: D7006A3E 02000149
	s_wait_alu depctr_va_vcc(0)                                // 000000001F30: BF88FF9D
	v_add_co_ci_u32_e64 v63, null, s1, v74, vcc_lo             // 000000001F34: D5207C3F 01AA9401
	v_cvt_i32_f32_e32 v99, v75                                 // 000000001F3C: 7EC6114B
	v_mul_f32_e32 v95, s40, v76                                // 000000001F40: 10BE9828
	v_add_co_u32 v75, vcc_lo, v62, s0                          // 000000001F44: D7006A4B 0200013E
	s_wait_alu depctr_va_vcc(0)                                // 000000001F4C: BF88FF9D
	v_add_co_ci_u32_e64 v76, null, s1, v63, vcc_lo             // 000000001F50: D5207C4C 01AA7E01
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 000000001F58: BF870193
	v_cvt_i32_f32_e32 v100, v95                                // 000000001F5C: 7EC8115F
	v_add_co_u32 v95, vcc_lo, v75, s0                          // 000000001F60: D7006A5F 0200014B
	s_wait_alu depctr_va_vcc(0)                                // 000000001F68: BF88FF9D
	s_delay_alu instid0(VALU_DEP_3)                            // 000000001F6C: BF870003
	v_add_co_ci_u32_e64 v96, null, s1, v76, vcc_lo             // 000000001F70: D5207C60 01AA9801
	v_cvt_i32_f32_e32 v48, v48                                 // 000000001F78: 7E601130
	s_clause 0x4                                               // 000000001F7C: BF850004
	global_store_b8 v[73:74], v97, off                         // 000000001F80: EE06007C 30800000 00000049
	global_store_b8 v[62:63], v98, off                         // 000000001F8C: EE06007C 31000000 0000003E
	global_store_b8 v[75:76], v99, off                         // 000000001F98: EE06007C 31800000 0000004B
	global_store_b8 v[95:96], v100, off                        // 000000001FA4: EE06007C 32000000 0000005F
	global_store_b8 v[56:57], v48, off offset:16               // 000000001FB0: EE06007C 18000000 00001038
	v_cvt_f32_i32_e32 v48, v49                                 // 000000001FBC: 7E600B31
	v_cvt_f32_i32_e32 v49, v50                                 // 000000001FC0: 7E620B32
	v_cvt_f32_i32_e32 v50, v51                                 // 000000001FC4: 7E640B33
	v_cvt_f32_i32_e32 v51, v52                                 // 000000001FC8: 7E660B34
	v_cvt_f32_i32_e32 v52, v53                                 // 000000001FCC: 7E680B35
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)// 000000001FD0: BF870194
	v_dual_mul_f32 v48, s40, v48 :: v_dual_mul_f32 v49, s40, v49// 000000001FD4: C8C66028 30306228
	v_dual_mul_f32 v50, s40, v50 :: v_dual_mul_f32 v51, s40, v51// 000000001FDC: C8C66428 32326628
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 000000001FE4: BF870193
	v_mul_f32_e32 v52, s40, v52                                // 000000001FE8: 10686828
	v_cvt_i32_f32_e32 v48, v48                                 // 000000001FEC: 7E601130
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)// 000000001FF0: BF870214
	v_cvt_i32_f32_e32 v49, v49                                 // 000000001FF4: 7E621131
	v_cvt_i32_f32_e32 v50, v50                                 // 000000001FF8: 7E641132
	v_cvt_i32_f32_e32 v51, v51                                 // 000000001FFC: 7E661133
	v_cvt_i32_f32_e32 v52, v52                                 // 000000002000: 7E681134
	s_clause 0x4                                               // 000000002004: BF850004
	global_store_b8 v[58:59], v48, off offset:16               // 000000002008: EE06007C 18000000 0000103A
	global_store_b8 v[60:61], v49, off offset:16               // 000000002014: EE06007C 18800000 0000103C
	global_store_b8 v[71:72], v50, off offset:16               // 000000002020: EE06007C 19000000 00001047
	global_store_b8 v[73:74], v51, off offset:16               // 00000000202C: EE06007C 19800000 00001049
	global_store_b8 v[62:63], v52, off offset:16               // 000000002038: EE06007C 1A000000 0000103E
	v_cvt_f32_i32_e32 v48, v54                                 // 000000002044: 7E600B36
	v_cvt_f32_i32_e32 v49, v55                                 // 000000002048: 7E620B37
	v_cvt_f32_i32_e32 v40, v40                                 // 00000000204C: 7E500B28
	v_cvt_f32_i32_e32 v41, v41                                 // 000000002050: 7E520B29
	v_cvt_f32_i32_e32 v42, v42                                 // 000000002054: 7E540B2A
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)// 000000002058: BF870194
	v_dual_mul_f32 v48, s40, v48 :: v_dual_mul_f32 v49, s40, v49// 00000000205C: C8C66028 30306228
	v_dual_mul_f32 v40, s40, v40 :: v_dual_mul_f32 v41, s40, v41// 000000002064: C8C65028 28285228
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 00000000206C: BF870193
	v_mul_f32_e32 v42, s40, v42                                // 000000002070: 10545428
	v_cvt_i32_f32_e32 v48, v48                                 // 000000002074: 7E601130
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)// 000000002078: BF870214
	v_cvt_i32_f32_e32 v49, v49                                 // 00000000207C: 7E621131
	v_cvt_i32_f32_e32 v40, v40                                 // 000000002080: 7E501128
	v_cvt_i32_f32_e32 v41, v41                                 // 000000002084: 7E521129
	v_cvt_i32_f32_e32 v42, v42                                 // 000000002088: 7E54112A
	s_clause 0x4                                               // 00000000208C: BF850004
	global_store_b8 v[75:76], v48, off offset:16               // 000000002090: EE06007C 18000000 0000104B
	global_store_b8 v[95:96], v49, off offset:16               // 00000000209C: EE06007C 18800000 0000105F
	global_store_b8 v[56:57], v40, off offset:32               // 0000000020A8: EE06007C 14000000 00002038
	global_store_b8 v[58:59], v41, off offset:32               // 0000000020B4: EE06007C 14800000 0000203A
	global_store_b8 v[60:61], v42, off offset:32               // 0000000020C0: EE06007C 15000000 0000203C
	v_cvt_f32_i32_e32 v40, v43                                 // 0000000020CC: 7E500B2B
	v_cvt_f32_i32_e32 v32, v32                                 // 0000000020D0: 7E400B20
	v_cvt_f32_i32_e32 v41, v44                                 // 0000000020D4: 7E520B2C
	v_cvt_f32_i32_e32 v44, v47                                 // 0000000020D8: 7E580B2F
	v_cvt_f32_i32_e32 v33, v33                                 // 0000000020DC: 7E420B21
	v_cvt_f32_i32_e32 v42, v45                                 // 0000000020E0: 7E540B2D
	v_cvt_f32_i32_e32 v34, v34                                 // 0000000020E4: 7E440B22
	v_cvt_f32_i32_e32 v43, v46                                 // 0000000020E8: 7E560B2E
	v_cvt_f32_i32_e32 v35, v35                                 // 0000000020EC: 7E460B23
	v_cvt_f32_i32_e32 v36, v36                                 // 0000000020F0: 7E480B24
	v_dual_mul_f32 v40, s40, v40 :: v_dual_mul_f32 v41, s40, v41// 0000000020F4: C8C65028 28285228
	s_delay_alu instid0(VALU_DEP_3)                            // 0000000020FC: BF870003
	v_dual_mul_f32 v32, s40, v32 :: v_dual_mul_f32 v35, s40, v35// 000000002100: C8C64028 20224628
	v_dual_mul_f32 v44, s40, v44 :: v_dual_mul_f32 v33, s40, v33// 000000002108: C8C65828 2C204228
	v_dual_mul_f32 v42, s40, v42 :: v_dual_mul_f32 v43, s40, v43// 000000002110: C8C65428 2A2A5628
	v_mul_f32_e32 v34, s40, v34                                // 000000002118: 10444428
	v_cvt_f32_i32_e32 v25, v25                                 // 00000000211C: 7E320B19
	v_mul_f32_e32 v36, s40, v36                                // 000000002120: 10484828
	v_cvt_i32_f32_e32 v40, v40                                 // 000000002124: 7E501128
	v_cvt_i32_f32_e32 v32, v32                                 // 000000002128: 7E401120
	v_cvt_i32_f32_e32 v41, v41                                 // 00000000212C: 7E521129
	v_cvt_i32_f32_e32 v33, v33                                 // 000000002130: 7E421121
	v_cvt_i32_f32_e32 v42, v42                                 // 000000002134: 7E54112A
	v_cvt_i32_f32_e32 v34, v34                                 // 000000002138: 7E441122
	v_cvt_i32_f32_e32 v43, v43                                 // 00000000213C: 7E56112B
	v_cvt_i32_f32_e32 v35, v35                                 // 000000002140: 7E461123
	v_cvt_i32_f32_e32 v44, v44                                 // 000000002144: 7E58112C
	v_mul_f32_e32 v25, s40, v25                                // 000000002148: 10323228
	v_cvt_i32_f32_e32 v36, v36                                 // 00000000214C: 7E481124
	s_clause 0x9                                               // 000000002150: BF850009
	global_store_b8 v[71:72], v40, off offset:32               // 000000002154: EE06007C 14000000 00002047
	global_store_b8 v[73:74], v41, off offset:32               // 000000002160: EE06007C 14800000 00002049
	global_store_b8 v[62:63], v42, off offset:32               // 00000000216C: EE06007C 15000000 0000203E
	global_store_b8 v[75:76], v43, off offset:32               // 000000002178: EE06007C 15800000 0000204B
	global_store_b8 v[95:96], v44, off offset:32               // 000000002184: EE06007C 16000000 0000205F
	global_store_b8 v[56:57], v32, off offset:48               // 000000002190: EE06007C 10000000 00003038
	global_store_b8 v[58:59], v33, off offset:48               // 00000000219C: EE06007C 10800000 0000303A
	global_store_b8 v[60:61], v34, off offset:48               // 0000000021A8: EE06007C 11000000 0000303C
	global_store_b8 v[71:72], v35, off offset:48               // 0000000021B4: EE06007C 11800000 00003047
	global_store_b8 v[73:74], v36, off offset:48               // 0000000021C0: EE06007C 12000000 00003049
	v_cvt_f32_i32_e32 v32, v37                                 // 0000000021CC: 7E400B25
	v_cvt_f32_i32_e32 v33, v38                                 // 0000000021D0: 7E420B26
	v_cvt_f32_i32_e32 v34, v39                                 // 0000000021D4: 7E440B27
	v_cvt_f32_i32_e32 v24, v24                                 // 0000000021D8: 7E300B18
	v_cvt_f32_i32_e32 v17, v17                                 // 0000000021DC: 7E220B11
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)// 0000000021E0: BF870214
	v_dual_mul_f32 v32, s40, v32 :: v_dual_mul_f32 v33, s40, v33// 0000000021E4: C8C64028 20204228
	v_mul_f32_e32 v34, s40, v34                                // 0000000021EC: 10444428
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 0000000021F0: BF870193
	v_dual_mul_f32 v24, s40, v24 :: v_dual_mul_f32 v17, s40, v17// 0000000021F4: C8C63028 18102228
	v_cvt_i32_f32_e32 v32, v32                                 // 0000000021FC: 7E401120
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)// 000000002200: BF870214
	v_cvt_i32_f32_e32 v33, v33                                 // 000000002204: 7E421121
	v_cvt_i32_f32_e32 v34, v34                                 // 000000002208: 7E441122
	s_delay_alu instid0(VALU_DEP_4)                            // 00000000220C: BF870004
	v_cvt_i32_f32_e32 v24, v24                                 // 000000002210: 7E301118
	v_cvt_i32_f32_e32 v25, v25                                 // 000000002214: 7E321119
	s_clause 0x4                                               // 000000002218: BF850004
	global_store_b8 v[62:63], v32, off offset:48               // 00000000221C: EE06007C 10000000 0000303E
	global_store_b8 v[75:76], v33, off offset:48               // 000000002228: EE06007C 10800000 0000304B
	global_store_b8 v[95:96], v34, off offset:48               // 000000002234: EE06007C 11000000 0000305F
	global_store_b8 v[56:57], v24, off offset:64               // 000000002240: EE06007C 0C000000 00004038
	global_store_b8 v[58:59], v25, off offset:64               // 00000000224C: EE06007C 0C800000 0000403A
	v_cvt_f32_i32_e32 v24, v26                                 // 000000002258: 7E300B1A
	v_cvt_f32_i32_e32 v25, v27                                 // 00000000225C: 7E320B1B
	v_cvt_f32_i32_e32 v26, v28                                 // 000000002260: 7E340B1C
	v_cvt_f32_i32_e32 v27, v29                                 // 000000002264: 7E360B1D
	v_cvt_f32_i32_e32 v28, v30                                 // 000000002268: 7E380B1E
	v_cvt_f32_i32_e32 v19, v19                                 // 00000000226C: 7E260B13
	v_dual_mul_f32 v24, s40, v24 :: v_dual_mul_f32 v25, s40, v25// 000000002270: C8C63028 18183228
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)// 000000002278: BF870194
	v_dual_mul_f32 v26, s40, v26 :: v_dual_mul_f32 v27, s40, v27// 00000000227C: C8C63428 1A1A3628
	v_dual_mul_f32 v28, s40, v28 :: v_dual_mul_f32 v19, s40, v19// 000000002284: C8C63828 1C122628
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)// 00000000228C: BF870213
	v_cvt_i32_f32_e32 v24, v24                                 // 000000002290: 7E301118
	v_cvt_i32_f32_e32 v25, v25                                 // 000000002294: 7E321119
	s_delay_alu instid0(VALU_DEP_4)                            // 000000002298: BF870004
	v_cvt_i32_f32_e32 v26, v26                                 // 00000000229C: 7E34111A
	v_cvt_i32_f32_e32 v27, v27                                 // 0000000022A0: 7E36111B
	v_cvt_i32_f32_e32 v28, v28                                 // 0000000022A4: 7E38111C
	s_clause 0x4                                               // 0000000022A8: BF850004
	global_store_b8 v[60:61], v24, off offset:64               // 0000000022AC: EE06007C 0C000000 0000403C
	global_store_b8 v[71:72], v25, off offset:64               // 0000000022B8: EE06007C 0C800000 00004047
	global_store_b8 v[73:74], v26, off offset:64               // 0000000022C4: EE06007C 0D000000 00004049
	global_store_b8 v[62:63], v27, off offset:64               // 0000000022D0: EE06007C 0D800000 0000403E
	global_store_b8 v[75:76], v28, off offset:64               // 0000000022DC: EE06007C 0E000000 0000404B
	v_cvt_f32_i32_e32 v24, v31                                 // 0000000022E8: 7E300B1F
	v_cvt_f32_i32_e32 v16, v16                                 // 0000000022EC: 7E200B10
	v_cvt_f32_i32_e32 v18, v18                                 // 0000000022F0: 7E240B12
	v_cvt_i32_f32_e32 v17, v17                                 // 0000000022F4: 7E221111
	v_cvt_i32_f32_e32 v19, v19                                 // 0000000022F8: 7E261113
	v_mul_f32_e32 v24, s40, v24                                // 0000000022FC: 10303028
	v_mul_f32_e32 v16, s40, v16                                // 000000002300: 10202028
	v_mul_f32_e32 v18, s40, v18                                // 000000002304: 10242428
	v_cvt_f32_i32_e32 v8, v8                                   // 000000002308: 7E100B08
	v_cvt_f32_i32_e32 v1, v1                                   // 00000000230C: 7E020B01
	v_cvt_i32_f32_e32 v24, v24                                 // 000000002310: 7E301118
	v_cvt_i32_f32_e32 v16, v16                                 // 000000002314: 7E201110
	v_cvt_i32_f32_e32 v18, v18                                 // 000000002318: 7E241112
	s_clause 0x4                                               // 00000000231C: BF850004
	global_store_b8 v[95:96], v24, off offset:64               // 000000002320: EE06007C 0C000000 0000405F
	global_store_b8 v[56:57], v16, off offset:80               // 00000000232C: EE06007C 08000000 00005038
	global_store_b8 v[58:59], v17, off offset:80               // 000000002338: EE06007C 08800000 0000503A
	global_store_b8 v[60:61], v18, off offset:80               // 000000002344: EE06007C 09000000 0000503C
	global_store_b8 v[71:72], v19, off offset:80               // 000000002350: EE06007C 09800000 00005047
	v_cvt_f32_i32_e32 v16, v20                                 // 00000000235C: 7E200B14
	v_cvt_f32_i32_e32 v17, v21                                 // 000000002360: 7E220B15
	v_cvt_f32_i32_e32 v18, v22                                 // 000000002364: 7E240B16
	v_cvt_f32_i32_e32 v19, v23                                 // 000000002368: 7E260B17
	v_mul_f32_e32 v8, s40, v8                                  // 00000000236C: 10101028
	v_dual_mul_f32 v16, s40, v16 :: v_dual_mul_f32 v1, s40, v1 // 000000002370: C8C62028 10000228
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)// 000000002378: BF870214
	v_dual_mul_f32 v17, s40, v17 :: v_dual_mul_f32 v18, s40, v18// 00000000237C: C8C62228 11122428
	v_mul_f32_e32 v19, s40, v19                                // 000000002384: 10262628
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)// 000000002388: BF870223
	v_cvt_i32_f32_e32 v16, v16                                 // 00000000238C: 7E201110
	v_cvt_i32_f32_e32 v8, v8                                   // 000000002390: 7E101108
	v_cvt_i32_f32_e32 v17, v17                                 // 000000002394: 7E221111
	v_cvt_i32_f32_e32 v18, v18                                 // 000000002398: 7E241112
	v_cvt_i32_f32_e32 v19, v19                                 // 00000000239C: 7E261113
	s_clause 0x4                                               // 0000000023A0: BF850004
	global_store_b8 v[73:74], v16, off offset:80               // 0000000023A4: EE06007C 08000000 00005049
	global_store_b8 v[62:63], v17, off offset:80               // 0000000023B0: EE06007C 08800000 0000503E
	global_store_b8 v[75:76], v18, off offset:80               // 0000000023BC: EE06007C 09000000 0000504B
	global_store_b8 v[95:96], v19, off offset:80               // 0000000023C8: EE06007C 09800000 0000505F
	global_store_b8 v[56:57], v8, off offset:96                // 0000000023D4: EE06007C 04000000 00006038
	v_cvt_f32_i32_e32 v8, v9                                   // 0000000023E0: 7E100B09
	v_cvt_f32_i32_e32 v9, v10                                  // 0000000023E4: 7E120B0A
	v_cvt_f32_i32_e32 v10, v11                                 // 0000000023E8: 7E140B0B
	v_cvt_f32_i32_e32 v11, v12                                 // 0000000023EC: 7E160B0C
	v_cvt_f32_i32_e32 v12, v13                                 // 0000000023F0: 7E180B0D
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)// 0000000023F4: BF870194
	v_dual_mul_f32 v8, s40, v8 :: v_dual_mul_f32 v9, s40, v9   // 0000000023F8: C8C61028 08081228
	v_dual_mul_f32 v10, s40, v10 :: v_dual_mul_f32 v11, s40, v11// 000000002400: C8C61428 0A0A1628
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 000000002408: BF870193
	v_mul_f32_e32 v12, s40, v12                                // 00000000240C: 10181828
	v_cvt_i32_f32_e32 v8, v8                                   // 000000002410: 7E101108
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)// 000000002414: BF870214
	v_cvt_i32_f32_e32 v9, v9                                   // 000000002418: 7E121109
	v_cvt_i32_f32_e32 v10, v10                                 // 00000000241C: 7E14110A
	v_cvt_i32_f32_e32 v11, v11                                 // 000000002420: 7E16110B
	v_cvt_i32_f32_e32 v12, v12                                 // 000000002424: 7E18110C
	s_clause 0x4                                               // 000000002428: BF850004
	global_store_b8 v[58:59], v8, off offset:96                // 00000000242C: EE06007C 04000000 0000603A
	global_store_b8 v[60:61], v9, off offset:96                // 000000002438: EE06007C 04800000 0000603C
	global_store_b8 v[71:72], v10, off offset:96               // 000000002444: EE06007C 05000000 00006047
	global_store_b8 v[73:74], v11, off offset:96               // 000000002450: EE06007C 05800000 00006049
	global_store_b8 v[62:63], v12, off offset:96               // 00000000245C: EE06007C 06000000 0000603E
	v_cvt_f32_i32_e32 v8, v14                                  // 000000002468: 7E100B0E
	v_cvt_f32_i32_e32 v9, v15                                  // 00000000246C: 7E120B0F
	v_cvt_f32_i32_e32 v0, v0                                   // 000000002470: 7E000B00
	v_cvt_f32_i32_e32 v2, v2                                   // 000000002474: 7E040B02
	v_cvt_i32_f32_e32 v1, v1                                   // 000000002478: 7E021101
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)// 00000000247C: BF870214
	v_dual_mul_f32 v8, s40, v8 :: v_dual_mul_f32 v9, s40, v9   // 000000002480: C8C61028 08081228
	v_mul_f32_e32 v0, s40, v0                                  // 000000002488: 10000028
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_3)// 00000000248C: BF8701A4
	v_mul_f32_e32 v2, s40, v2                                  // 000000002490: 10040428
	s_add_co_i32 s24, s36, s24                                 // 000000002494: 81181824
	v_cvt_i32_f32_e32 v8, v8                                   // 000000002498: 7E101108
	v_cvt_i32_f32_e32 v9, v9                                   // 00000000249C: 7E121109
	v_cvt_i32_f32_e32 v0, v0                                   // 0000000024A0: 7E001100
	v_cvt_i32_f32_e32 v2, v2                                   // 0000000024A4: 7E041102
	s_clause 0x4                                               // 0000000024A8: BF850004
	global_store_b8 v[75:76], v8, off offset:96                // 0000000024AC: EE06007C 04000000 0000604B
	global_store_b8 v[95:96], v9, off offset:96                // 0000000024B8: EE06007C 04800000 0000605F
	global_store_b8 v[56:57], v0, off offset:112               // 0000000024C4: EE06007C 00000000 00007038
	global_store_b8 v[58:59], v1, off offset:112               // 0000000024D0: EE06007C 00800000 0000703A
	global_store_b8 v[60:61], v2, off offset:112               // 0000000024DC: EE06007C 01000000 0000703C
	v_cvt_f32_i32_e32 v0, v3                                   // 0000000024E8: 7E000B03
	v_cvt_f32_i32_e32 v1, v4                                   // 0000000024EC: 7E020B04
	v_cvt_f32_i32_e32 v2, v5                                   // 0000000024F0: 7E040B05
	v_cvt_f32_i32_e32 v3, v6                                   // 0000000024F4: 7E060B06
	v_cvt_f32_i32_e32 v4, v7                                   // 0000000024F8: 7E080B07
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)// 0000000024FC: BF870194
	v_dual_mul_f32 v0, s40, v0 :: v_dual_mul_f32 v1, s40, v1   // 000000002500: C8C60028 00000228
	v_dual_mul_f32 v2, s40, v2 :: v_dual_mul_f32 v3, s40, v3   // 000000002508: C8C60428 02020628
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)// 000000002510: BF870193
	v_mul_f32_e32 v4, s40, v4                                  // 000000002514: 10080828
	v_cvt_i32_f32_e32 v0, v0                                   // 000000002518: 7E001100
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)// 00000000251C: BF870214
	v_cvt_i32_f32_e32 v1, v1                                   // 000000002520: 7E021101
	v_cvt_i32_f32_e32 v2, v2                                   // 000000002524: 7E041102
	v_cvt_i32_f32_e32 v3, v3                                   // 000000002528: 7E061103
	v_cvt_i32_f32_e32 v4, v4                                   // 00000000252C: 7E081104
	s_clause 0x4                                               // 000000002530: BF850004
	global_store_b8 v[71:72], v0, off offset:112               // 000000002534: EE06007C 00000000 00007047
	global_store_b8 v[73:74], v1, off offset:112               // 000000002540: EE06007C 00800000 00007049
	global_store_b8 v[62:63], v2, off offset:112               // 00000000254C: EE06007C 01000000 0000703E
	global_store_b8 v[75:76], v3, off offset:112               // 000000002558: EE06007C 01800000 0000704B
	global_store_b8 v[95:96], v4, off offset:112               // 000000002564: EE06007C 02000000 0000705F
	s_wait_loadcnt 0x0                                         // 000000002570: BFC00000
	s_wait_storecnt 0x0                                        // 000000002574: BFC10000
	s_barrier_signal -1                                        // 000000002578: BE804EC1
	s_wait_alu depctr_sa_sdst(0)                               // 00000000257C: BF88FF9E
	s_cmp_ge_i32 s24, s22                                      // 000000002580: BF031618
	s_barrier_wait 0xffff                                      // 000000002584: BF94FFFF
	global_inv scope:SCOPE_SE                                  // 000000002588: EE0AC07C 00040000 00000000
	s_cbranch_scc1 536                                         // 000000002594: BFA20218 <moe_v19+0x12f8>
	s_ashr_i32 s25, s24, 31                                    // 000000002598: 86199F18
	s_wait_alu depctr_sa_sdst(0)                               // 00000000259C: BF88FF9E
	s_lshl_b64 s[26:27], s[24:25], 2                           // 0000000025A0: 849A8218
	s_wait_alu depctr_sa_sdst(0)                               // 0000000025A4: BF88FF9E
	s_add_nc_u64 s[28:29], s[10:11], s[26:27]                  // 0000000025A8: A99C1A0A
	s_load_b32 s25, s[28:29], 0x0                              // 0000000025AC: F400064E F8000000
	s_add_nc_u64 s[28:29], s[8:9], s[26:27]                    // 0000000025B4: A99C1A08
	s_add_nc_u64 s[26:27], s[12:13], s[26:27]                  // 0000000025B8: A99A1A0C
	s_load_b32 s14, s[28:29], 0x0                              // 0000000025BC: F400038E F8000000
	s_load_b32 s26, s[26:27], 0x0                              // 0000000025C4: F400068D F8000000
	s_wait_kmcnt 0x0                                           // 0000000025CC: BFC70000
	s_mul_i32 s28, s25, s21                                    // 0000000025D0: 961C1519
	s_wait_alu depctr_sa_sdst(0)                               // 0000000025D4: BF88FF9E
	s_ashr_i32 s29, s28, 31                                    // 0000000025D8: 861D9F1C
	s_wait_alu depctr_sa_sdst(0)                               // 0000000025DC: BF88FF9E
	s_lshl_b64 s[34:35], s[28:29], 2                           // 0000000025E0: 84A2821C
	s_wait_alu depctr_sa_sdst(0)                               // 0000000025E4: BF88FF9E
	s_add_nc_u64 s[30:31], s[4:5], s[34:35]                    // 0000000025E8: A99E2204
	s_and_saveexec_b32 s27, s2                                 // 0000000025EC: BE9B2002
	s_cbranch_execz 18                                         // 0000000025F0: BFA50012 <moe_v19+0xb3c>
	s_wait_alu depctr_sa_sdst(0)                               // 0000000025F4: BF88FF9E
	v_add_co_u32 v0, vcc_lo, s30, v67                          // 0000000025F8: D7006A00 0202861E
	s_wait_alu depctr_va_vcc(0)                                // 000000002600: BF88FF9D
	v_add_co_ci_u32_e64 v1, null, s31, v68, vcc_lo             // 000000002604: D5207C01 01AA881F
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 00000000260C: BF870122
	v_add_co_u32 v0, vcc_lo, v0, v93                           // 000000002610: D7006A00 0202BB00
	s_wait_alu depctr_va_vcc(0)                                // 000000002618: BF88FF9D
	v_add_co_ci_u32_e64 v1, null, 0, v1, vcc_lo                // 00000000261C: D5207C01 01AA0280
	global_load_b64 v[0:1], v[0:1], off                        // 000000002624: EE05407C 00000000 00000000
	s_wait_loadcnt 0x0                                         // 000000002630: BFC00000
	ds_store_b64 v81, v[0:1]                                   // 000000002634: D9340000 00000051
	s_wait_alu depctr_sa_sdst(0)                               // 00000000263C: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s27                             // 000000002640: 8C7E1B7E
	s_mul_i32 s28, s23, s14                                    // 000000002644: 961C0E17
	s_ashr_i32 s27, s26, 31                                    // 000000002648: 861B9F1A
	s_wait_alu depctr_sa_sdst(0)                               // 00000000264C: BF88FF9E
	s_ashr_i32 s29, s28, 31                                    // 000000002650: 861D9F1C
	s_lshl_b64 s[42:43], s[26:27], 2                           // 000000002654: 84AA821A
	s_wait_alu depctr_sa_sdst(0)                               // 000000002658: BF88FF9E
	s_lshl_b64 s[28:29], s[28:29], 2                           // 00000000265C: 849C821C
	s_wait_alu depctr_sa_sdst(0)                               // 000000002660: BF88FF9E
	s_add_nc_u64 s[28:29], s[6:7], s[28:29]                    // 000000002664: A99C1C06
	s_wait_alu depctr_sa_sdst(0)                               // 000000002668: BF88FF9E
	s_add_nc_u64 s[28:29], s[28:29], s[42:43]                  // 00000000266C: A99C2A1C
	s_and_saveexec_b32 s14, s3                                 // 000000002670: BE8E2003
	s_cbranch_execz 22                                         // 000000002674: BFA50016 <moe_v19+0xbd0>
	s_wait_alu depctr_sa_sdst(0)                               // 000000002678: BF88FF9E
	v_add_co_u32 v0, vcc_lo, s28, v69                          // 00000000267C: D7006A00 02028A1C
	s_wait_alu depctr_va_vcc(0)                                // 000000002684: BF88FF9D
	v_add_co_ci_u32_e64 v1, null, s29, v70, vcc_lo             // 000000002688: D5207C01 01AA8C1D
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000002690: BF870122
	v_add_co_u32 v2, vcc_lo, v0, s18                           // 000000002694: D7006A02 02002500
	s_wait_alu depctr_va_vcc(0)                                // 00000000269C: BF88FF9D
	v_add_co_ci_u32_e64 v3, null, s19, v1, vcc_lo              // 0000000026A0: D5207C03 01AA0213
	s_clause 0x1                                               // 0000000026A8: BF850001
	global_load_b32 v0, v[0:1], off                            // 0000000026AC: EE05007C 00000000 00000000
	global_load_b32 v1, v[2:3], off                            // 0000000026B8: EE05007C 00000001 00000002
	s_wait_loadcnt 0x0                                         // 0000000026C4: BFC00000
	ds_store_b64 v90, v[0:1] offset:6144                       // 0000000026C8: D9341800 0000005A
	s_or_b32 exec_lo, exec_lo, s14                             // 0000000026D0: 8C7E0E7E
	s_wait_dscnt 0x0                                           // 0000000026D4: BFC60000
	s_barrier_signal -1                                        // 0000000026D8: BE804EC1
	s_and_not1_b32 vcc_lo, exec_lo, s33                        // 0000000026DC: 916A217E
	s_barrier_wait 0xffff                                      // 0000000026E0: BF94FFFF
	global_inv scope:SCOPE_SE                                  // 0000000026E4: EE0AC07C 00040000 00000000
	s_wait_alu depctr_sa_sdst(0)                               // 0000000026F0: BF88FF9E
	s_cbranch_vccnz 64899                                      // 0000000026F4: BFA4FD83 <moe_v19+0x204>
	s_and_not1_b32 vcc_lo, exec_lo, s37                        // 0000000026F8: 916A257E
	s_wait_alu depctr_sa_sdst(0)                               // 0000000026FC: BF88FF9E
	s_cbranch_vccnz 253                                        // 000000002700: BFA400FD <moe_v19+0xff8>
	v_dual_mov_b32 v56, 0 :: v_dual_mov_b32 v73, v66           // 000000002704: CA100080 38480142
	v_add_co_u32 v71, vcc_lo, v88, s34                         // 00000000270C: D7006A47 02004558
	s_wait_alu depctr_va_vcc(0)                                // 000000002714: BF88FF9D
	v_add_co_ci_u32_e64 v72, null, s35, v89, vcc_lo            // 000000002718: D5207C48 01AAB223
	s_delay_alu instid0(VALU_DEP_3)                            // 000000002720: BF870003
	v_dual_mov_b32 v75, v65 :: v_dual_mov_b32 v58, v56         // 000000002724: CA100141 4B3A0138
	v_dual_mov_b32 v57, v56 :: v_dual_mov_b32 v60, v56         // 00000000272C: CA100138 393C0138
	v_dual_mov_b32 v59, v56 :: v_dual_mov_b32 v62, v56         // 000000002734: CA100138 3B3E0138
	v_dual_mov_b32 v61, v56 :: v_dual_mov_b32 v48, v56         // 00000000273C: CA100138 3D300138
	v_dual_mov_b32 v63, v56 :: v_dual_mov_b32 v50, v56         // 000000002744: CA100138 3F320138
	v_dual_mov_b32 v49, v56 :: v_dual_mov_b32 v52, v56         // 00000000274C: CA100138 31340138
	v_dual_mov_b32 v51, v56 :: v_dual_mov_b32 v54, v56         // 000000002754: CA100138 33360138
	v_dual_mov_b32 v53, v56 :: v_dual_mov_b32 v40, v56         // 00000000275C: CA100138 35280138
	v_dual_mov_b32 v55, v56 :: v_dual_mov_b32 v42, v56         // 000000002764: CA100138 372A0138
	v_dual_mov_b32 v41, v56 :: v_dual_mov_b32 v44, v56         // 00000000276C: CA100138 292C0138
	v_dual_mov_b32 v43, v56 :: v_dual_mov_b32 v46, v56         // 000000002774: CA100138 2B2E0138
	v_dual_mov_b32 v45, v56 :: v_dual_mov_b32 v32, v56         // 00000000277C: CA100138 2D200138
	v_dual_mov_b32 v47, v56 :: v_dual_mov_b32 v34, v56         // 000000002784: CA100138 2F220138
	v_dual_mov_b32 v33, v56 :: v_dual_mov_b32 v36, v56         // 00000000278C: CA100138 21240138
	v_dual_mov_b32 v35, v56 :: v_dual_mov_b32 v38, v56         // 000000002794: CA100138 23260138
	v_dual_mov_b32 v37, v56 :: v_dual_mov_b32 v24, v56         // 00000000279C: CA100138 25180138
	v_dual_mov_b32 v39, v56 :: v_dual_mov_b32 v26, v56         // 0000000027A4: CA100138 271A0138
	v_dual_mov_b32 v25, v56 :: v_dual_mov_b32 v28, v56         // 0000000027AC: CA100138 191C0138
	v_dual_mov_b32 v27, v56 :: v_dual_mov_b32 v30, v56         // 0000000027B4: CA100138 1B1E0138
	v_dual_mov_b32 v29, v56 :: v_dual_mov_b32 v16, v56         // 0000000027BC: CA100138 1D100138
	v_dual_mov_b32 v31, v56 :: v_dual_mov_b32 v18, v56         // 0000000027C4: CA100138 1F120138
	v_dual_mov_b32 v17, v56 :: v_dual_mov_b32 v20, v56         // 0000000027CC: CA100138 11140138
	v_dual_mov_b32 v19, v56 :: v_dual_mov_b32 v22, v56         // 0000000027D4: CA100138 13160138
	v_dual_mov_b32 v21, v56 :: v_dual_mov_b32 v8, v56          // 0000000027DC: CA100138 15080138
	v_dual_mov_b32 v23, v56 :: v_dual_mov_b32 v10, v56         // 0000000027E4: CA100138 170A0138
	v_dual_mov_b32 v9, v56 :: v_dual_mov_b32 v12, v56          // 0000000027EC: CA100138 090C0138
	v_dual_mov_b32 v11, v56 :: v_dual_mov_b32 v14, v56         // 0000000027F4: CA100138 0B0E0138
	v_dual_mov_b32 v13, v56 :: v_dual_mov_b32 v0, v56          // 0000000027FC: CA100138 0D000138
	v_dual_mov_b32 v15, v56 :: v_dual_mov_b32 v2, v56          // 000000002804: CA100138 0F020138
	v_dual_mov_b32 v1, v56 :: v_dual_mov_b32 v4, v56           // 00000000280C: CA100138 01040138
	v_dual_mov_b32 v3, v56 :: v_dual_mov_b32 v6, v56           // 000000002814: CA100138 03060138
	v_mov_b32_e32 v5, v56                                      // 00000000281C: 7E0A0338
	v_mov_b32_e32 v7, v56                                      // 000000002820: 7E0E0338
	s_mov_b32 s14, 4                                           // 000000002824: BE8E0084
	s_mov_b32 s27, s38                                         // 000000002828: BE9B0026
	s_branch 20                                                // 00000000282C: BFA00014 <moe_v19+0xd80>
	s_wait_alu depctr_sa_sdst(0)                               // 000000002830: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s34                             // 000000002834: 8C7E227E
	s_wait_loadcnt_dscnt 0x0                                   // 000000002838: BFC80000
	s_barrier_signal -1                                        // 00000000283C: BE804EC1
	v_add_co_u32 v71, vcc_lo, v71, 32                          // 000000002840: D7006A47 02014147
	s_wait_alu depctr_va_vcc(0)                                // 000000002848: BF88FF9D
	v_add_co_ci_u32_e64 v72, null, 0, v72, vcc_lo              // 00000000284C: D5207C48 01AA9080
	v_add_nc_u32_e32 v75, s41, v75                             // 000000002854: 4A969629
	v_add_nc_u32_e32 v73, s41, v73                             // 000000002858: 4A929229
	s_add_co_i32 s27, s27, -2                                  // 00000000285C: 811BC21B
	s_add_co_i32 s14, s14, 8                                   // 000000002860: 810E880E
	s_wait_alu depctr_sa_sdst(0)                               // 000000002864: BF88FF9E
	s_cmp_eq_u32 s27, 0                                        // 000000002868: BF06801B
	s_barrier_wait 0xffff                                      // 00000000286C: BF94FFFF
	global_inv scope:SCOPE_SE                                  // 000000002870: EE0AC07C 00040000 00000000
	s_cbranch_scc1 151                                         // 00000000287C: BFA20097 <moe_v19+0xfdc>
	ds_load_b64 v[111:112], v92                                // 000000002880: D9D80000 6F00005C
	ds_load_2addr_b64 v[95:98], v94 offset1:16                 // 000000002888: D9DC1000 5F00005E
	ds_load_2addr_b64 v[99:102], v94 offset0:32 offset1:48     // 000000002890: D9DC3020 6300005E
	ds_load_2addr_b64 v[103:106], v94 offset0:64 offset1:80    // 000000002898: D9DC5040 6700005E
	ds_load_2addr_b64 v[107:110], v94 offset0:96 offset1:112   // 0000000028A0: D9DC7060 6B00005E
	s_cmp_ge_i32 s14, s21                                      // 0000000028A8: BF03150E
	s_wait_dscnt 0x3                                           // 0000000028AC: BFC60003
	v_wmma_i32_16x16x32_iu4 v[56:63], v[111:112], v[95:96], v[56:63] neg_lo:[1,1,0]// 0000000028B0: CC4A4038 7CE2BF6F
	v_wmma_i32_16x16x32_iu4 v[48:55], v[111:112], v[97:98], v[48:55] neg_lo:[1,1,0]// 0000000028B8: CC4A4030 7CC2C36F
	s_wait_dscnt 0x2                                           // 0000000028C0: BFC60002
	v_wmma_i32_16x16x32_iu4 v[40:47], v[111:112], v[99:100], v[40:47] neg_lo:[1,1,0]// 0000000028C4: CC4A4028 7CA2C76F
	v_wmma_i32_16x16x32_iu4 v[32:39], v[111:112], v[101:102], v[32:39] neg_lo:[1,1,0]// 0000000028CC: CC4A4020 7C82CB6F
	s_wait_dscnt 0x1                                           // 0000000028D4: BFC60001
	v_wmma_i32_16x16x32_iu4 v[24:31], v[111:112], v[103:104], v[24:31] neg_lo:[1,1,0]// 0000000028D8: CC4A4018 7C62CF6F
	v_wmma_i32_16x16x32_iu4 v[16:23], v[111:112], v[105:106], v[16:23] neg_lo:[1,1,0]// 0000000028E0: CC4A4010 7C42D36F
	s_wait_dscnt 0x0                                           // 0000000028E8: BFC60000
	v_wmma_i32_16x16x32_iu4 v[8:15], v[111:112], v[107:108], v[8:15] neg_lo:[1,1,0]// 0000000028EC: CC4A4008 7C22D76F
	v_wmma_i32_16x16x32_iu4 v[0:7], v[111:112], v[109:110], v[0:7] neg_lo:[1,1,0]// 0000000028F4: CC4A4000 7C02DB6F
	s_cbranch_scc1 39                                          // 0000000028FC: BFA20027 <moe_v19+0xe9c>
	s_and_saveexec_b32 s34, s2                                 // 000000002900: BEA22002
	s_cbranch_execz 6                                          // 000000002904: BFA50006 <moe_v19+0xe20>
	global_load_b64 v[95:96], v[71:72], off                    // 000000002908: EE05407C 0000005F 00000047
	s_wait_loadcnt 0x0                                         // 000000002914: BFC00000
	ds_store_b64 v81, v[95:96] offset:8192                     // 000000002918: D9342000 00005F51
	s_wait_alu depctr_sa_sdst(0)                               // 000000002920: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s34                             // 000000002924: 8C7E227E
	s_and_saveexec_b32 s34, s3                                 // 000000002928: BEA22003
	s_cbranch_execz 25                                         // 00000000292C: BFA50019 <moe_v19+0xe94>
	v_ashrrev_i32_e32 v76, 31, v75                             // 000000002930: 3498969F
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000002934: BF870091
	v_lshlrev_b64_e32 v[95:96], 2, v[75:76]                    // 000000002938: 3EBE9682
	v_add_co_u32 v95, vcc_lo, s28, v95                         // 00000000293C: D7006A5F 0202BE1C
	s_wait_alu depctr_va_vcc(0)                                // 000000002944: BF88FF9D
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000002948: BF870112
	v_add_co_ci_u32_e64 v96, null, s29, v96, vcc_lo            // 00000000294C: D5207C60 01AAC01D
	v_add_co_u32 v97, vcc_lo, v95, s18                         // 000000002954: D7006A61 0200255F
	s_wait_alu depctr_va_vcc(0)                                // 00000000295C: BF88FF9D
	s_delay_alu instid0(VALU_DEP_2)                            // 000000002960: BF870002
	v_add_co_ci_u32_e64 v98, null, s19, v96, vcc_lo            // 000000002964: D5207C62 01AAC013
	s_clause 0x1                                               // 00000000296C: BF850001
	global_load_b32 v95, v[95:96], off                         // 000000002970: EE05007C 0000005F 0000005F
	global_load_b32 v96, v[97:98], off                         // 00000000297C: EE05007C 00000060 00000061
	s_wait_loadcnt 0x0                                         // 000000002988: BFC00000
	ds_store_b64 v90, v[95:96] offset:14336                    // 00000000298C: D9343800 00005F5A
	s_wait_alu depctr_sa_sdst(0)                               // 000000002994: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s34                             // 000000002998: 8C7E227E
	s_wait_loadcnt_dscnt 0x0                                   // 00000000299C: BFC80000
	s_barrier_signal -1                                        // 0000000029A0: BE804EC1
	v_add_nc_u32_e32 v74, 0x3800, v91                          // 0000000029A4: 4A94B6FF 00003800
	s_add_co_i32 s34, s14, 4                                   // 0000000029AC: 8122840E
	s_wait_alu depctr_sa_sdst(0)                               // 0000000029B0: BF88FF9E
	s_cmp_ge_i32 s34, s21                                      // 0000000029B4: BF031522
	s_barrier_wait 0xffff                                      // 0000000029B8: BF94FFFF
	global_inv scope:SCOPE_SE                                  // 0000000029BC: EE0AC07C 00040000 00000000
	ds_load_b64 v[111:112], v92 offset:8192                    // 0000000029C8: D9D82000 6F00005C
	ds_load_2addr_b64 v[95:98], v74 offset1:16                 // 0000000029D0: D9DC1000 5F00004A
	ds_load_2addr_b64 v[99:102], v74 offset0:32 offset1:48     // 0000000029D8: D9DC3020 6300004A
	ds_load_2addr_b64 v[103:106], v74 offset0:64 offset1:80    // 0000000029E0: D9DC5040 6700004A
	ds_load_2addr_b64 v[107:110], v74 offset0:96 offset1:112   // 0000000029E8: D9DC7060 6B00004A
	s_wait_dscnt 0x3                                           // 0000000029F0: BFC60003
	v_wmma_i32_16x16x32_iu4 v[56:63], v[111:112], v[95:96], v[56:63] neg_lo:[1,1,0]// 0000000029F4: CC4A4038 7CE2BF6F
	v_wmma_i32_16x16x32_iu4 v[48:55], v[111:112], v[97:98], v[48:55] neg_lo:[1,1,0]// 0000000029FC: CC4A4030 7CC2C36F
	s_wait_dscnt 0x2                                           // 000000002A04: BFC60002
	v_wmma_i32_16x16x32_iu4 v[40:47], v[111:112], v[99:100], v[40:47] neg_lo:[1,1,0]// 000000002A08: CC4A4028 7CA2C76F
	v_wmma_i32_16x16x32_iu4 v[32:39], v[111:112], v[101:102], v[32:39] neg_lo:[1,1,0]// 000000002A10: CC4A4020 7C82CB6F
	s_wait_dscnt 0x1                                           // 000000002A18: BFC60001
	v_wmma_i32_16x16x32_iu4 v[24:31], v[111:112], v[103:104], v[24:31] neg_lo:[1,1,0]// 000000002A1C: CC4A4018 7C62CF6F
	v_wmma_i32_16x16x32_iu4 v[16:23], v[111:112], v[105:106], v[16:23] neg_lo:[1,1,0]// 000000002A24: CC4A4010 7C42D36F
	s_wait_dscnt 0x0                                           // 000000002A2C: BFC60000
	v_wmma_i32_16x16x32_iu4 v[8:15], v[111:112], v[107:108], v[8:15] neg_lo:[1,1,0]// 000000002A30: CC4A4008 7C22D76F
	v_wmma_i32_16x16x32_iu4 v[0:7], v[111:112], v[109:110], v[0:7] neg_lo:[1,1,0]// 000000002A38: CC4A4000 7C02DB6F
	s_cbranch_scc1 65405                                       // 000000002A40: BFA2FF7D <moe_v19+0xd38>
	s_and_saveexec_b32 s34, s2                                 // 000000002A44: BEA22002
	s_cbranch_execz 6                                          // 000000002A48: BFA50006 <moe_v19+0xf64>
	global_load_b64 v[95:96], v[71:72], off offset:16          // 000000002A4C: EE05407C 0000005F 00001047
	s_wait_loadcnt 0x0                                         // 000000002A58: BFC00000
	ds_store_b64 v81, v[95:96]                                 // 000000002A5C: D9340000 00005F51
	s_wait_alu depctr_sa_sdst(0)                               // 000000002A64: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s34                             // 000000002A68: 8C7E227E
	s_and_saveexec_b32 s34, s3                                 // 000000002A6C: BEA22003
	s_cbranch_execz 65391                                      // 000000002A70: BFA5FF6F <moe_v19+0xd30>
	v_ashrrev_i32_e32 v74, 31, v73                             // 000000002A74: 3494929F
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000002A78: BF870091
	v_lshlrev_b64_e32 v[95:96], 2, v[73:74]                    // 000000002A7C: 3EBE9282
	v_add_co_u32 v95, vcc_lo, s28, v95                         // 000000002A80: D7006A5F 0202BE1C
	s_wait_alu depctr_va_vcc(0)                                // 000000002A88: BF88FF9D
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000002A8C: BF870112
	v_add_co_ci_u32_e64 v96, null, s29, v96, vcc_lo            // 000000002A90: D5207C60 01AAC01D
	v_add_co_u32 v97, vcc_lo, v95, s18                         // 000000002A98: D7006A61 0200255F
	s_wait_alu depctr_va_vcc(0)                                // 000000002AA0: BF88FF9D
	s_delay_alu instid0(VALU_DEP_2)                            // 000000002AA4: BF870002
	v_add_co_ci_u32_e64 v98, null, s19, v96, vcc_lo            // 000000002AA8: D5207C62 01AAC013
	s_clause 0x1                                               // 000000002AB0: BF850001
	global_load_b32 v95, v[95:96], off                         // 000000002AB4: EE05007C 0000005F 0000005F
	global_load_b32 v96, v[97:98], off                         // 000000002AC0: EE05007C 00000060 00000061
	s_wait_loadcnt 0x0                                         // 000000002ACC: BFC00000
	ds_store_b64 v90, v[95:96] offset:6144                     // 000000002AD0: D9341800 00005F5A
	s_branch 65365                                             // 000000002AD8: BFA0FF55 <moe_v19+0xd30>
	s_add_co_i32 s14, s14, -4                                  // 000000002ADC: 810EC40E
	s_mov_b32 s27, s39                                         // 000000002AE0: BE9B0027
	s_wait_alu depctr_sa_sdst(0)                               // 000000002AE4: BF88FF9E
	s_and_b32 vcc_lo, exec_lo, s27                             // 000000002AE8: 8B6A1B7E
	s_wait_alu depctr_sa_sdst(0)                               // 000000002AEC: BF88FF9E
	s_cbranch_vccz 64709                                       // 000000002AF0: BFA3FCC5 <moe_v19+0x308>
	s_branch 67                                                // 000000002AF4: BFA00043 <moe_v19+0x1104>
	v_mov_b32_e32 v7, 0                                        // 000000002AF8: 7E0E0280
	s_mov_b32 s14, 0                                           // 000000002AFC: BE8E0080
	s_delay_alu instid0(VALU_DEP_1)                            // 000000002B00: BF870001
	v_dual_mov_b32 v6, v7 :: v_dual_mov_b32 v5, v7             // 000000002B04: CA100107 06040107
	v_dual_mov_b32 v4, v7 :: v_dual_mov_b32 v3, v7             // 000000002B0C: CA100107 04020107
	v_dual_mov_b32 v2, v7 :: v_dual_mov_b32 v1, v7             // 000000002B14: CA100107 02000107
	v_dual_mov_b32 v0, v7 :: v_dual_mov_b32 v15, v7            // 000000002B1C: CA100107 000E0107
	v_dual_mov_b32 v14, v7 :: v_dual_mov_b32 v13, v7           // 000000002B24: CA100107 0E0C0107
	v_dual_mov_b32 v12, v7 :: v_dual_mov_b32 v11, v7           // 000000002B2C: CA100107 0C0A0107
	v_dual_mov_b32 v10, v7 :: v_dual_mov_b32 v9, v7            // 000000002B34: CA100107 0A080107
	v_dual_mov_b32 v8, v7 :: v_dual_mov_b32 v23, v7            // 000000002B3C: CA100107 08160107
	v_dual_mov_b32 v22, v7 :: v_dual_mov_b32 v21, v7           // 000000002B44: CA100107 16140107
	v_dual_mov_b32 v20, v7 :: v_dual_mov_b32 v19, v7           // 000000002B4C: CA100107 14120107
	v_dual_mov_b32 v18, v7 :: v_dual_mov_b32 v17, v7           // 000000002B54: CA100107 12100107
	v_dual_mov_b32 v16, v7 :: v_dual_mov_b32 v31, v7           // 000000002B5C: CA100107 101E0107
	v_dual_mov_b32 v30, v7 :: v_dual_mov_b32 v29, v7           // 000000002B64: CA100107 1E1C0107
	v_dual_mov_b32 v28, v7 :: v_dual_mov_b32 v27, v7           // 000000002B6C: CA100107 1C1A0107
	v_dual_mov_b32 v26, v7 :: v_dual_mov_b32 v25, v7           // 000000002B74: CA100107 1A180107
	v_dual_mov_b32 v24, v7 :: v_dual_mov_b32 v39, v7           // 000000002B7C: CA100107 18260107
	v_dual_mov_b32 v38, v7 :: v_dual_mov_b32 v37, v7           // 000000002B84: CA100107 26240107
	v_dual_mov_b32 v36, v7 :: v_dual_mov_b32 v35, v7           // 000000002B8C: CA100107 24220107
	v_dual_mov_b32 v34, v7 :: v_dual_mov_b32 v33, v7           // 000000002B94: CA100107 22200107
	v_dual_mov_b32 v32, v7 :: v_dual_mov_b32 v47, v7           // 000000002B9C: CA100107 202E0107
	v_dual_mov_b32 v46, v7 :: v_dual_mov_b32 v45, v7           // 000000002BA4: CA100107 2E2C0107
	v_dual_mov_b32 v44, v7 :: v_dual_mov_b32 v43, v7           // 000000002BAC: CA100107 2C2A0107
	v_dual_mov_b32 v42, v7 :: v_dual_mov_b32 v41, v7           // 000000002BB4: CA100107 2A280107
	v_dual_mov_b32 v40, v7 :: v_dual_mov_b32 v55, v7           // 000000002BBC: CA100107 28360107
	v_dual_mov_b32 v54, v7 :: v_dual_mov_b32 v53, v7           // 000000002BC4: CA100107 36340107
	v_dual_mov_b32 v52, v7 :: v_dual_mov_b32 v51, v7           // 000000002BCC: CA100107 34320107
	v_dual_mov_b32 v50, v7 :: v_dual_mov_b32 v49, v7           // 000000002BD4: CA100107 32300107
	v_dual_mov_b32 v48, v7 :: v_dual_mov_b32 v63, v7           // 000000002BDC: CA100107 303E0107
	v_dual_mov_b32 v62, v7 :: v_dual_mov_b32 v61, v7           // 000000002BE4: CA100107 3E3C0107
	v_dual_mov_b32 v60, v7 :: v_dual_mov_b32 v59, v7           // 000000002BEC: CA100107 3C3A0107
	v_dual_mov_b32 v58, v7 :: v_dual_mov_b32 v57, v7           // 000000002BF4: CA100107 3A380107
	v_mov_b32_e32 v56, v7                                      // 000000002BFC: 7E700307
	s_cbranch_execz 64641                                      // 000000002C00: BFA5FC81 <moe_v19+0x308>
	s_lshl_b32 s27, s14, 11                                    // 000000002C04: 841B8B0E
	s_add_co_i32 s34, s14, 4                                   // 000000002C08: 8122840E
	s_wait_alu depctr_sa_sdst(0)                               // 000000002C0C: BF88FF9E
	s_and_b32 s27, s27, 0x2000                                 // 000000002C10: 8B1BFF1B 00002000
	s_wait_alu depctr_sa_sdst(0)                               // 000000002C18: BF88FF9E
	s_add_co_i32 s27, s27, 0                                   // 000000002C1C: 811B801B
	s_cmp_ge_i32 s34, s21                                      // 000000002C20: BF031522
	s_wait_alu depctr_sa_sdst(0)                               // 000000002C24: BF88FF9E
	v_add3_u32 v71, s27, v85, v86                              // 000000002C28: D6550047 055AAA1B
	v_add3_u32 v72, s27, v83, v84                              // 000000002C30: D6550048 0552A61B
	s_delay_alu instid0(VALU_DEP_2)                            // 000000002C38: BF870002
	v_add_nc_u32_e32 v103, 0x1800, v71                         // 000000002C3C: 4ACE8EFF 00001800
	ds_load_b64 v[75:76], v72                                  // 000000002C44: D9D80000 4B000048
	ds_load_2addr_b64 v[71:74], v103 offset1:16                // 000000002C4C: D9DC1000 47000067
	ds_load_2addr_b64 v[95:98], v103 offset0:32 offset1:48     // 000000002C54: D9DC3020 5F000067
	ds_load_2addr_b64 v[99:102], v103 offset0:64 offset1:80    // 000000002C5C: D9DC5040 63000067
	ds_load_2addr_b64 v[103:106], v103 offset0:96 offset1:112  // 000000002C64: D9DC7060 67000067
	s_wait_dscnt 0x3                                           // 000000002C6C: BFC60003
	v_wmma_i32_16x16x32_iu4 v[56:63], v[75:76], v[71:72], v[56:63] neg_lo:[1,1,0]// 000000002C70: CC4A4038 7CE28F4B
	v_wmma_i32_16x16x32_iu4 v[48:55], v[75:76], v[73:74], v[48:55] neg_lo:[1,1,0]// 000000002C78: CC4A4030 7CC2934B
	s_wait_dscnt 0x2                                           // 000000002C80: BFC60002
	v_wmma_i32_16x16x32_iu4 v[40:47], v[75:76], v[95:96], v[40:47] neg_lo:[1,1,0]// 000000002C84: CC4A4028 7CA2BF4B
	v_wmma_i32_16x16x32_iu4 v[32:39], v[75:76], v[97:98], v[32:39] neg_lo:[1,1,0]// 000000002C8C: CC4A4020 7C82C34B
	s_wait_dscnt 0x1                                           // 000000002C94: BFC60001
	v_wmma_i32_16x16x32_iu4 v[24:31], v[75:76], v[99:100], v[24:31] neg_lo:[1,1,0]// 000000002C98: CC4A4018 7C62C74B
	v_wmma_i32_16x16x32_iu4 v[16:23], v[75:76], v[101:102], v[16:23] neg_lo:[1,1,0]// 000000002CA0: CC4A4010 7C42CB4B
	s_wait_dscnt 0x0                                           // 000000002CA8: BFC60000
	v_wmma_i32_16x16x32_iu4 v[8:15], v[75:76], v[103:104], v[8:15] neg_lo:[1,1,0]// 000000002CAC: CC4A4008 7C22CF4B
	v_wmma_i32_16x16x32_iu4 v[0:7], v[75:76], v[105:106], v[0:7] neg_lo:[1,1,0]// 000000002CB4: CC4A4000 7C02D34B
	s_cbranch_scc1 71                                          // 000000002CBC: BFA20047 <moe_v19+0x12dc>
	s_not_b32 s27, s14                                         // 000000002CC0: BE9B1E0E
	s_wait_alu depctr_sa_sdst(0)                               // 000000002CC4: BF88FF9E
	s_lshl_b32 s27, s27, 11                                    // 000000002CC8: 841B8B1B
	s_wait_alu depctr_sa_sdst(0)                               // 000000002CCC: BF88FF9E
	s_and_b32 s27, s27, 0x2000                                 // 000000002CD0: 8B1BFF1B 00002000
	s_wait_alu depctr_sa_sdst(0)                               // 000000002CD8: BF88FF9E
	s_add_co_i32 s27, s27, 0                                   // 000000002CDC: 811B801B
	s_and_saveexec_b32 s35, s2                                 // 000000002CE0: BEA32002
	s_cbranch_execz 25                                         // 000000002CE4: BFA50019 <moe_v19+0x124c>
	v_add_co_u32 v71, vcc_lo, s30, v67                         // 000000002CE8: D7006A47 0202861E
	s_wait_alu depctr_va_vcc(0)                                // 000000002CF0: BF88FF9D
	v_add_co_ci_u32_e64 v72, null, s31, v68, vcc_lo            // 000000002CF4: D5207C48 01AA881F
	s_lshl_b64 s[30:31], s[14:15], 2                           // 000000002CFC: 849E820E
	v_add_co_u32 v71, vcc_lo, v71, v93                         // 000000002D00: D7006A47 0202BB47
	s_wait_alu depctr_va_vcc(0)                                // 000000002D08: BF88FF9D
	v_add_co_ci_u32_e64 v72, null, 0, v72, vcc_lo              // 000000002D0C: D5207C48 01AA9080
	s_wait_alu depctr_sa_sdst(0)                               // 000000002D14: BF88FF9E
	v_add3_u32 v73, s27, v80, v93                              // 000000002D18: D6550049 0576A01B
	v_add_co_u32 v71, vcc_lo, v71, s30                         // 000000002D20: D7006A47 02003D47
	s_wait_alu depctr_va_vcc(0)                                // 000000002D28: BF88FF9D
	v_add_co_ci_u32_e64 v72, null, s31, v72, vcc_lo            // 000000002D2C: D5207C48 01AA901F
	global_load_b64 v[71:72], v[71:72], off offset:16          // 000000002D34: EE05407C 00000047 00001047
	s_wait_loadcnt 0x0                                         // 000000002D40: BFC00000
	ds_store_b64 v73, v[71:72]                                 // 000000002D44: D9340000 00004749
	s_wait_alu depctr_sa_sdst(0)                               // 000000002D4C: BF88FF9E
	s_or_b32 exec_lo, exec_lo, s35                             // 000000002D50: 8C7E237E
	s_and_saveexec_b32 s14, s3                                 // 000000002D54: BE8E2003
	s_cbranch_execz 31                                         // 000000002D58: BFA5001F <moe_v19+0x12d8>
	v_or_b32_e32 v71, s34, v78                                 // 000000002D5C: 388E9C22
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000002D60: BF870091
	v_mad_co_u64_u32 v[71:72], null, v71, s20, v[64:65]        // 000000002D64: D6FE7C47 05002947
	v_ashrrev_i32_e32 v72, 31, v71                             // 000000002D6C: 34908E9F
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000002D70: BF870091
	v_lshlrev_b64_e32 v[71:72], 2, v[71:72]                    // 000000002D74: 3E8E8E82
	v_add_co_u32 v71, vcc_lo, s28, v71                         // 000000002D78: D7006A47 02028E1C
	s_wait_alu depctr_va_vcc(0)                                // 000000002D80: BF88FF9D
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000002D84: BF870112
	v_add_co_ci_u32_e64 v72, null, s29, v72, vcc_lo            // 000000002D88: D5207C48 01AA901D
	v_add_co_u32 v73, vcc_lo, v71, s18                         // 000000002D90: D7006A49 02002547
	s_wait_alu depctr_va_vcc(0)                                // 000000002D98: BF88FF9D
	s_delay_alu instid0(VALU_DEP_2)                            // 000000002D9C: BF870002
	v_add_co_ci_u32_e64 v74, null, s19, v72, vcc_lo            // 000000002DA0: D5207C4A 01AA9013
	s_clause 0x1                                               // 000000002DA8: BF850001
	global_load_b32 v71, v[71:72], off                         // 000000002DAC: EE05007C 00000047 00000047
	global_load_b32 v72, v[73:74], off                         // 000000002DB8: EE05007C 00000048 00000049
	v_add3_u32 v73, s27, v79, v82                              // 000000002DC4: D6550049 054A9E1B
	s_wait_loadcnt 0x0                                         // 000000002DCC: BFC00000
	ds_store_b64 v73, v[71:72] offset:6144                     // 000000002DD0: D9341800 00004749
	s_or_b32 exec_lo, exec_lo, s14                             // 000000002DD8: 8C7E0E7E
	s_wait_loadcnt_dscnt 0x0                                   // 000000002DDC: BFC80000
	s_barrier_signal -1                                        // 000000002DE0: BE804EC1
	s_barrier_wait 0xffff                                      // 000000002DE4: BF94FFFF
	global_inv scope:SCOPE_SE                                  // 000000002DE8: EE0AC07C 00040000 00000000
	s_branch 64516                                             // 000000002DF4: BFA0FC04 <moe_v19+0x308>
	s_endpgm                                                   // 000000002DF8: BFB00000
	s_code_end                                                 // 000000002DFC: BF9F0000
	s_code_end                                                 // 000000002E00: BF9F0000
	s_code_end                                                 // 000000002E04: BF9F0000
	s_code_end                                                 // 000000002E08: BF9F0000
	s_code_end                                                 // 000000002E0C: BF9F0000
	s_code_end                                                 // 000000002E10: BF9F0000
	s_code_end                                                 // 000000002E14: BF9F0000
	s_code_end                                                 // 000000002E18: BF9F0000
	s_code_end                                                 // 000000002E1C: BF9F0000
	s_code_end                                                 // 000000002E20: BF9F0000
	s_code_end                                                 // 000000002E24: BF9F0000
	s_code_end                                                 // 000000002E28: BF9F0000
	s_code_end                                                 // 000000002E2C: BF9F0000
	s_code_end                                                 // 000000002E30: BF9F0000
	s_code_end                                                 // 000000002E34: BF9F0000
	s_code_end                                                 // 000000002E38: BF9F0000
	s_code_end                                                 // 000000002E3C: BF9F0000
	s_code_end                                                 // 000000002E40: BF9F0000
	s_code_end                                                 // 000000002E44: BF9F0000
	s_code_end                                                 // 000000002E48: BF9F0000
	s_code_end                                                 // 000000002E4C: BF9F0000
	s_code_end                                                 // 000000002E50: BF9F0000
	s_code_end                                                 // 000000002E54: BF9F0000
	s_code_end                                                 // 000000002E58: BF9F0000
	s_code_end                                                 // 000000002E5C: BF9F0000
	s_code_end                                                 // 000000002E60: BF9F0000
	s_code_end                                                 // 000000002E64: BF9F0000
	s_code_end                                                 // 000000002E68: BF9F0000
	s_code_end                                                 // 000000002E6C: BF9F0000
	s_code_end                                                 // 000000002E70: BF9F0000
	s_code_end                                                 // 000000002E74: BF9F0000
	s_code_end                                                 // 000000002E78: BF9F0000
	s_code_end                                                 // 000000002E7C: BF9F0000
	s_code_end                                                 // 000000002E80: BF9F0000
	s_code_end                                                 // 000000002E84: BF9F0000
	s_code_end                                                 // 000000002E88: BF9F0000
	s_code_end                                                 // 000000002E8C: BF9F0000
	s_code_end                                                 // 000000002E90: BF9F0000
	s_code_end                                                 // 000000002E94: BF9F0000
	s_code_end                                                 // 000000002E98: BF9F0000
	s_code_end                                                 // 000000002E9C: BF9F0000
	s_code_end                                                 // 000000002EA0: BF9F0000
	s_code_end                                                 // 000000002EA4: BF9F0000
	s_code_end                                                 // 000000002EA8: BF9F0000
	s_code_end                                                 // 000000002EAC: BF9F0000
	s_code_end                                                 // 000000002EB0: BF9F0000
	s_code_end                                                 // 000000002EB4: BF9F0000
	s_code_end                                                 // 000000002EB8: BF9F0000
	s_code_end                                                 // 000000002EBC: BF9F0000
	s_code_end                                                 // 000000002EC0: BF9F0000
	s_code_end                                                 // 000000002EC4: BF9F0000
	s_code_end                                                 // 000000002EC8: BF9F0000
	s_code_end                                                 // 000000002ECC: BF9F0000
	s_code_end                                                 // 000000002ED0: BF9F0000
	s_code_end                                                 // 000000002ED4: BF9F0000
	s_code_end                                                 // 000000002ED8: BF9F0000
	s_code_end                                                 // 000000002EDC: BF9F0000
	s_code_end                                                 // 000000002EE0: BF9F0000
	s_code_end                                                 // 000000002EE4: BF9F0000
	s_code_end                                                 // 000000002EE8: BF9F0000
	s_code_end                                                 // 000000002EEC: BF9F0000
	s_code_end                                                 // 000000002EF0: BF9F0000
	s_code_end                                                 // 000000002EF4: BF9F0000
	s_code_end                                                 // 000000002EF8: BF9F0000
	s_code_end                                                 // 000000002EFC: BF9F0000
	s_code_end                                                 // 000000002F00: BF9F0000
	s_code_end                                                 // 000000002F04: BF9F0000
	s_code_end                                                 // 000000002F08: BF9F0000
	s_code_end                                                 // 000000002F0C: BF9F0000
	s_code_end                                                 // 000000002F10: BF9F0000
	s_code_end                                                 // 000000002F14: BF9F0000
	s_code_end                                                 // 000000002F18: BF9F0000
	s_code_end                                                 // 000000002F1C: BF9F0000
	s_code_end                                                 // 000000002F20: BF9F0000
	s_code_end                                                 // 000000002F24: BF9F0000
	s_code_end                                                 // 000000002F28: BF9F0000
	s_code_end                                                 // 000000002F2C: BF9F0000
	s_code_end                                                 // 000000002F30: BF9F0000
	s_code_end                                                 // 000000002F34: BF9F0000
	s_code_end                                                 // 000000002F38: BF9F0000
	s_code_end                                                 // 000000002F3C: BF9F0000
	s_code_end                                                 // 000000002F40: BF9F0000
	s_code_end                                                 // 000000002F44: BF9F0000
	s_code_end                                                 // 000000002F48: BF9F0000
	s_code_end                                                 // 000000002F4C: BF9F0000
	s_code_end                                                 // 000000002F50: BF9F0000
	s_code_end                                                 // 000000002F54: BF9F0000
	s_code_end                                                 // 000000002F58: BF9F0000
	s_code_end                                                 // 000000002F5C: BF9F0000
	s_code_end                                                 // 000000002F60: BF9F0000
	s_code_end                                                 // 000000002F64: BF9F0000
	s_code_end                                                 // 000000002F68: BF9F0000
	s_code_end                                                 // 000000002F6C: BF9F0000
	s_code_end                                                 // 000000002F70: BF9F0000
	s_code_end                                                 // 000000002F74: BF9F0000
	s_code_end                                                 // 000000002F78: BF9F0000
	s_code_end                                                 // 000000002F7C: BF9F0000
