# R3 item 8 / P2-d: Delta_fit on Kvasir-Capsule and brain MRI

Commit: c3df6ed

Verdict (PRIMARY p_order, mahalanobis_l2): point: fails (Kvasir K_seg gap median +0.0408 vs brain median +0.0443); robust: fails (median Kvasir lower bound -0.0064 vs median brain upper bound +0.0706).

Precommit: results/r3/8/p2d/PRECOMMIT.json (FINAL). Pilot plan: results/r3/8/p2d/pilot_plan.md.

## Pilot (ceiling, frozen DINOv2-B, no Delta)

| dataset | candidate | Maha AUROC seen | Maha AUROC unseen | kNN seen | kNN unseen | skipped (> 0.97) |
|---|---|---|---|---|---|---|
| kvasir_capsule | Angiectasia | 0.5170 | 0.5526 | 0.4623 | 0.5018 | no |
| kvasir_capsule | Erosion | 0.5120 | 0.4805 | 0.4457 | 0.4185 | no |
| brain | meningioma | 0.7331 | 0.6927 | 0.6515 | 0.6698 | no |
| brain | pituitary | 0.6087 | 0.5364 | 0.4642 | 0.4507 | no |
| brain | glioma | 0.8824 | 0.8306 | 0.8326 | 0.8075 | no |

Primary: Kvasir-Capsule Angiectasia (first non-skipped candidate); brain meningioma (first non-skipped candidate). Only the primary candidate is trained and scored (the PRECOMMIT GPU estimate covers one candidate per dataset); secondary candidates are not run. The pilot's seen - unseen differences are disclosed in the table and not interpreted.

## Gates

- G_audit_sanity: Kvasir-Capsule three-way parse agrees (True, split vs archive videos True); brain counts match the README: True.
- G_leak: Kvasir-Capsule K_lit 1.0; brain 0.9885 (share of seen candidates with a fit image in the group).
- G_competence (preregistered: CNN final train loss <= 50% of epoch-1 loss and seen macro one-vs-rest AUROC >= 0.75; FM probe seen macro AUROC >= 0.70). Post hoc variant (AUROC only), as in P2-a, shown beside it.

| dataset / variant | run | kind | ID macro AUROC seen | unseen | ID accuracy seen | loss epoch 1 -> final | preregistered | post hoc |
|---|---|---|---|---|---|---|---|---|
| Kvasir-Capsule K_lit | resnet50_s42 | cnn | 0.9993 | 0.8949 | 0.9893 | 0.4510 -> 0.0174 | pass | pass |
| Kvasir-Capsule K_lit | resnet50_s43 | cnn | 0.9989 | 0.8729 | 0.9818 | 0.4508 -> 0.0193 | pass | pass |
| Kvasir-Capsule K_lit | convnext_tiny_s42 | cnn | 0.9995 | 0.9122 | 0.9873 | 0.3901 -> 0.0093 | pass | pass |
| Kvasir-Capsule K_lit | convnext_tiny_s43 | cnn | 0.9989 | 0.9194 | 0.9831 | 0.4213 -> 0.0104 | pass | pass |
| Kvasir-Capsule K_lit | dinov2_vitb14 | probe | 0.9935 | 0.8753 | 0.9762 | n/a | pass | pass |
| Kvasir-Capsule K_lit | biomedclip | probe | 0.9797 | 0.9024 | 0.9236 | n/a | pass | pass |
| Kvasir-Capsule K_seg | resnet50_s42 | cnn | 0.9443 | 0.8463 | 0.8073 | 0.3926 -> 0.0112 | pass | pass |
| Kvasir-Capsule K_seg | resnet50_s43 | cnn | 0.9558 | 0.8585 | 0.7911 | 0.3948 -> 0.0154 | pass | pass |
| Kvasir-Capsule K_seg | convnext_tiny_s42 | cnn | 0.9527 | 0.9209 | 0.7635 | 0.3661 -> 0.0065 | pass | pass |
| Kvasir-Capsule K_seg | convnext_tiny_s43 | cnn | 0.9451 | 0.9320 | 0.7940 | 0.3595 -> 0.0077 | pass | pass |
| Kvasir-Capsule K_seg | dinov2_vitb14 | probe | 0.9089 | 0.8881 | 0.7643 | n/a | pass | pass |
| Kvasir-Capsule K_seg | biomedclip | probe | 0.9141 | 0.8662 | 0.7712 | n/a | pass | pass |
| brain MRI | resnet50_s42 | cnn | 0.9996 | 0.9999 | 0.9913 | 0.2526 -> 0.0007 | pass | pass |
| brain MRI | resnet50_s43 | cnn | 0.9998 | 0.9997 | 0.9796 | 0.2472 -> 0.0020 | pass | pass |
| brain MRI | convnext_tiny_s42 | cnn | 0.9996 | 1.0000 | 0.9738 | 0.3058 -> 0.0001 | pass | pass |
| brain MRI | convnext_tiny_s43 | cnn | 1.0000 | 1.0000 | 0.9971 | 0.3630 -> 0.0006 | pass | pass |
| brain MRI | dinov2_vitb14 | probe | 0.9998 | 0.9990 | 0.9913 | n/a | pass | pass |
| brain MRI | biomedclip | probe | 1.0000 | 0.9976 | 0.9883 | n/a | pass | pass |

Gate-passing backbones (preregistered): Kvasir-Capsule K_lit: resnet50, convnext_tiny, dinov2_vitb14, biomedclip; Kvasir-Capsule K_seg: resnet50, convnext_tiny, dinov2_vitb14, biomedclip; brain MRI: resnet50, convnext_tiny, dinov2_vitb14, biomedclip.
Post hoc (AUROC only): Kvasir-Capsule K_lit: resnet50, convnext_tiny, dinov2_vitb14, biomedclip; Kvasir-Capsule K_seg: resnet50, convnext_tiny, dinov2_vitb14, biomedclip; brain MRI: resnet50, convnext_tiny, dinov2_vitb14, biomedclip.

## Predictions (preregistered gate)

- p_order: point: fails (Kvasir K_seg gap median +0.0408 vs brain median +0.0443); robust: fails (median Kvasir lower bound -0.0064 vs median brain upper bound +0.0706)
- p1_kvasir: fails (0/4 gate-passing backbones with Delta_fit > 0.05; robust: fails, 0 with jackknife CI > 0)
- p2_brain: fails (1/4 gate-passing backbones with Delta_fit > 0.05; robust: fails, 1 with jackknife CI > 0)
- p_order_knn: point: holds (Kvasir K_seg gap median +0.0713 vs brain median +0.0174); robust: fails (median Kvasir lower bound -0.0035 vs median brain upper bound +0.0316)
- p1_kvasir_knn: fails (1/4 gate-passing backbones with Delta_fit > 0.05; robust: fails, 0 with jackknife CI > 0)
- p2_brain_knn: fails (0/4 gate-passing backbones with Delta_fit > 0.05; robust: fails, 0 with jackknife CI > 0)
- p_gap_knn: point: holds (no-gap median +0.0738 vs gap median +0.0713); robust: fails (median lower bound of the paired difference -0.0008)

## Verdict rule (per dataset)

- Kvasir-Capsule (K_seg with gap), mahalanobis_l2: fails (0/4 gate-passing backbones with Delta_fit > 0.05; robust: fails, 0 with jackknife CI > 0)
- brain MRI, mahalanobis_l2: fails (1/4 gate-passing backbones with Delta_fit > 0.05; robust: fails, 1 with jackknife CI > 0)

## Cells

Jackknife (new, primary) and fixed-score cluster bootstrap (old, reference only) 95% CIs side by side. n_groups_fit = fit groups per F2 fold; K = 2.

| dataset / variant | run | scorer | statistic | estimate | jackknife CI | bootstrap CI (ref) | n_groups_fit | K | d | ID acc seen | flags |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Kvasir-Capsule K_lit | resnet50_s42 | mahalanobis_l2 | leaky | +0.8081 | [+0.5830, +1.0333] | [+0.7526, +0.9558] | [13, 14] | 2 | 2048 | 0.9893 | below_chance, a_gap_underpowered |
| Kvasir-Capsule K_lit | resnet50_s42 | mahalanobis_l2 | F2 | +0.2878 | [-0.2560, +0.8317] | [+0.1789, +0.6790] | [13, 14] | 2 | 2048 | 0.9893 | below_chance, a_gap_underpowered |
| Kvasir-Capsule K_lit | resnet50_s42 | mahalanobis_l2 | delta_fit | +0.5203 | [+0.1906, +0.8500] | [+0.2676, +0.6071] | [13, 14] | 2 | 2048 | 0.9893 | below_chance, a_gap_underpowered |
| Kvasir-Capsule K_lit | resnet50_s42 | mahalanobis_l2 | truth | +0.3239 | [-0.2812, +0.9290] | [+0.1554, +0.7688] | [13, 14] | 2 | 2048 | 0.9893 | below_chance, a_gap_underpowered |
| Kvasir-Capsule K_lit | resnet50_s42 | mahalanobis_l2 | a_gap | +0.4843 | [+0.0840, +0.8845] | [+0.1781, +0.6330] | [13, 14] | 2 | 2048 | 0.9893 | below_chance, a_gap_underpowered |
| Kvasir-Capsule K_lit | resnet50_s42 | mahalanobis_l2 | a_gap_cm | +0.4826 | [+0.0783, +0.8870] | [+0.1784, +0.6316] | [13, 14] | 2 | 2048 | 0.9893 | below_chance, a_gap_underpowered |
| Kvasir-Capsule K_lit | resnet50_s42 | knn_mean_cosine | leaky | +0.8697 | [+0.7211, +1.0184] | [+0.8312, +0.9572] | [13, 14] | 2 | 2048 | 0.9893 | below_chance, a_gap_underpowered |
| Kvasir-Capsule K_lit | resnet50_s42 | knn_mean_cosine | F2 | +0.4960 | [+0.1100, +0.8820] | [+0.3999, +0.7679] | [13, 14] | 2 | 2048 | 0.9893 | below_chance, a_gap_underpowered |
| Kvasir-Capsule K_lit | resnet50_s42 | knn_mean_cosine | delta_fit | +0.3737 | [+0.1223, +0.6252] | [+0.1857, +0.4519] | [13, 14] | 2 | 2048 | 0.9893 | below_chance, a_gap_underpowered |
| Kvasir-Capsule K_lit | resnet50_s42 | knn_mean_cosine | truth | +0.4385 | [+0.0101, +0.8668] | [+0.2784, +0.7712] | [13, 14] | 2 | 2048 | 0.9893 | below_chance, a_gap_underpowered |
| Kvasir-Capsule K_lit | resnet50_s42 | knn_mean_cosine | a_gap | +0.4313 | [+0.1264, +0.7361] | [+0.1732, +0.5840] | [13, 14] | 2 | 2048 | 0.9893 | below_chance, a_gap_underpowered |
| Kvasir-Capsule K_lit | resnet50_s42 | knn_mean_cosine | a_gap_cm | +0.4311 | [+0.1201, +0.7421] | [+0.1728, +0.5824] | [13, 14] | 2 | 2048 | 0.9893 | below_chance, a_gap_underpowered |
| Kvasir-Capsule K_lit | resnet50_s43 | mahalanobis_l2 | leaky | +0.8263 | [+0.6406, +1.0121] | [+0.7691, +0.9568] | [13, 14] | 2 | 2048 | 0.9818 | below_chance, a_gap_underpowered |
| Kvasir-Capsule K_lit | resnet50_s43 | mahalanobis_l2 | F2 | +0.3706 | [-0.1254, +0.8667] | [+0.2553, +0.7289] | [13, 14] | 2 | 2048 | 0.9818 | below_chance, a_gap_underpowered |
| Kvasir-Capsule K_lit | resnet50_s43 | mahalanobis_l2 | delta_fit | +0.4557 | [+0.1346, +0.7769] | [+0.2205, +0.5439] | [13, 14] | 2 | 2048 | 0.9818 | below_chance, a_gap_underpowered |
| Kvasir-Capsule K_lit | resnet50_s43 | mahalanobis_l2 | truth | +0.3237 | [-0.2637, +0.9112] | [+0.1674, +0.7673] | [13, 14] | 2 | 2048 | 0.9818 | below_chance, a_gap_underpowered |
| Kvasir-Capsule K_lit | resnet50_s43 | mahalanobis_l2 | a_gap | +0.5026 | [+0.0860, +0.9192] | [+0.1812, +0.6418] | [13, 14] | 2 | 2048 | 0.9818 | below_chance, a_gap_underpowered |
| Kvasir-Capsule K_lit | resnet50_s43 | mahalanobis_l2 | a_gap_cm | +0.5008 | [+0.0827, +0.9189] | [+0.1794, +0.6384] | [13, 14] | 2 | 2048 | 0.9818 | below_chance, a_gap_underpowered |
| Kvasir-Capsule K_lit | resnet50_s43 | knn_mean_cosine | leaky | +0.8843 | [+0.7950, +0.9736] | [+0.8509, +0.9409] | [13, 14] | 2 | 2048 | 0.9818 | below_chance, a_gap_underpowered |
| Kvasir-Capsule K_lit | resnet50_s43 | knn_mean_cosine | F2 | +0.5677 | [+0.3384, +0.7971] | [+0.4878, +0.7257] | [13, 14] | 2 | 2048 | 0.9818 | below_chance, a_gap_underpowered |
| Kvasir-Capsule K_lit | resnet50_s43 | knn_mean_cosine | delta_fit | +0.3166 | [+0.1605, +0.4726] | [+0.2063, +0.3863] | [13, 14] | 2 | 2048 | 0.9818 | below_chance, a_gap_underpowered |
| Kvasir-Capsule K_lit | resnet50_s43 | knn_mean_cosine | truth | +0.4420 | [+0.1406, +0.7434] | [+0.3087, +0.6809] | [13, 14] | 2 | 2048 | 0.9818 | below_chance, a_gap_underpowered |
| Kvasir-Capsule K_lit | resnet50_s43 | knn_mean_cosine | a_gap | +0.4423 | [+0.2049, +0.6797] | [+0.2411, +0.5742] | [13, 14] | 2 | 2048 | 0.9818 | below_chance, a_gap_underpowered |
| Kvasir-Capsule K_lit | resnet50_s43 | knn_mean_cosine | a_gap_cm | +0.4424 | [+0.1998, +0.6850] | [+0.2393, +0.5741] | [13, 14] | 2 | 2048 | 0.9818 | below_chance, a_gap_underpowered |
| Kvasir-Capsule K_lit | convnext_tiny_s42 | mahalanobis_l2 | leaky | +0.8226 | [+0.6135, +1.0317] | [+0.7624, +0.9755] | [13, 14] | 2 | 768 | 0.9873 | a_gap_underpowered |
| Kvasir-Capsule K_lit | convnext_tiny_s42 | mahalanobis_l2 | F2 | +0.5633 | [+0.1648, +0.9619] | [+0.4458, +0.8618] | [13, 14] | 2 | 768 | 0.9873 | a_gap_underpowered |
| Kvasir-Capsule K_lit | convnext_tiny_s42 | mahalanobis_l2 | delta_fit | +0.2592 | [+0.0588, +0.4597] | [+0.1063, +0.3324] | [13, 14] | 2 | 768 | 0.9873 | a_gap_underpowered |
| Kvasir-Capsule K_lit | convnext_tiny_s42 | mahalanobis_l2 | truth | +0.5397 | [+0.0985, +0.9810] | [+0.3003, +0.8779] | [13, 14] | 2 | 768 | 0.9873 | a_gap_underpowered |
| Kvasir-Capsule K_lit | convnext_tiny_s42 | mahalanobis_l2 | a_gap | +0.2828 | [-0.0085, +0.5742] | [+0.0651, +0.5066] | [13, 14] | 2 | 768 | 0.9873 | a_gap_underpowered |
| Kvasir-Capsule K_lit | convnext_tiny_s42 | mahalanobis_l2 | a_gap_cm | +0.2831 | [+0.0056, +0.5605] | [+0.0656, +0.5058] | [13, 14] | 2 | 768 | 0.9873 | a_gap_underpowered |
| Kvasir-Capsule K_lit | convnext_tiny_s42 | knn_mean_cosine | leaky | +0.8671 | [+0.7082, +1.0261] | [+0.8196, +0.9788] | [13, 14] | 2 | 768 | 0.9873 | a_gap_underpowered |
| Kvasir-Capsule K_lit | convnext_tiny_s42 | knn_mean_cosine | F2 | +0.6007 | [+0.1858, +1.0155] | [+0.4934, +0.8922] | [13, 14] | 2 | 768 | 0.9873 | a_gap_underpowered |
| Kvasir-Capsule K_lit | convnext_tiny_s42 | knn_mean_cosine | delta_fit | +0.2665 | [+0.0043, +0.5287] | [+0.0816, +0.3414] | [13, 14] | 2 | 768 | 0.9873 | a_gap_underpowered |
| Kvasir-Capsule K_lit | convnext_tiny_s42 | knn_mean_cosine | truth | +0.5357 | [+0.1179, +0.9535] | [+0.3060, +0.8600] | [13, 14] | 2 | 768 | 0.9873 | a_gap_underpowered |
| Kvasir-Capsule K_lit | convnext_tiny_s42 | knn_mean_cosine | a_gap | +0.3314 | [+0.0277, +0.6351] | [+0.0974, +0.5499] | [13, 14] | 2 | 768 | 0.9873 | a_gap_underpowered |
| Kvasir-Capsule K_lit | convnext_tiny_s42 | knn_mean_cosine | a_gap_cm | +0.3326 | [+0.0327, +0.6325] | [+0.0959, +0.5495] | [13, 14] | 2 | 768 | 0.9873 | a_gap_underpowered |
| Kvasir-Capsule K_lit | convnext_tiny_s43 | mahalanobis_l2 | leaky | +0.8064 | [+0.5676, +1.0452] | [+0.7373, +0.9825] | [13, 14] | 2 | 768 | 0.9831 | a_gap_underpowered |
| Kvasir-Capsule K_lit | convnext_tiny_s43 | mahalanobis_l2 | F2 | +0.5820 | [+0.1482, +1.0158] | [+0.4652, +0.9115] | [13, 14] | 2 | 768 | 0.9831 | a_gap_underpowered |
| Kvasir-Capsule K_lit | convnext_tiny_s43 | mahalanobis_l2 | delta_fit | +0.2244 | [+0.0217, +0.4271] | [+0.0673, +0.2887] | [13, 14] | 2 | 768 | 0.9831 | a_gap_underpowered |
| Kvasir-Capsule K_lit | convnext_tiny_s43 | mahalanobis_l2 | truth | +0.5570 | [+0.0685, +1.0455] | [+0.3377, +0.9194] | [13, 14] | 2 | 768 | 0.9831 | a_gap_underpowered |
| Kvasir-Capsule K_lit | convnext_tiny_s43 | mahalanobis_l2 | a_gap | +0.2494 | [-0.0423, +0.5411] | [+0.0459, +0.4527] | [13, 14] | 2 | 768 | 0.9831 | a_gap_underpowered |
| Kvasir-Capsule K_lit | convnext_tiny_s43 | mahalanobis_l2 | a_gap_cm | +0.2574 | [-0.0298, +0.5446] | [+0.0476, +0.4578] | [13, 14] | 2 | 768 | 0.9831 | a_gap_underpowered |
| Kvasir-Capsule K_lit | convnext_tiny_s43 | knn_mean_cosine | leaky | +0.8429 | [+0.6518, +1.0340] | [+0.7851, +0.9830] | [13, 14] | 2 | 768 | 0.9831 | a_gap_underpowered |
| Kvasir-Capsule K_lit | convnext_tiny_s43 | knn_mean_cosine | F2 | +0.6131 | [+0.2039, +1.0223] | [+0.5057, +0.9135] | [13, 14] | 2 | 768 | 0.9831 | a_gap_underpowered |
| Kvasir-Capsule K_lit | convnext_tiny_s43 | knn_mean_cosine | delta_fit | +0.2298 | [+0.0018, +0.4579] | [+0.0659, +0.2966] | [13, 14] | 2 | 768 | 0.9831 | a_gap_underpowered |
| Kvasir-Capsule K_lit | convnext_tiny_s43 | knn_mean_cosine | truth | +0.5474 | [+0.1041, +0.9908] | [+0.3400, +0.8893] | [13, 14] | 2 | 768 | 0.9831 | a_gap_underpowered |
| Kvasir-Capsule K_lit | convnext_tiny_s43 | knn_mean_cosine | a_gap | +0.2955 | [+0.0064, +0.5846] | [+0.0764, +0.4878] | [13, 14] | 2 | 768 | 0.9831 | a_gap_underpowered |
| Kvasir-Capsule K_lit | convnext_tiny_s43 | knn_mean_cosine | a_gap_cm | +0.3033 | [+0.0105, +0.5961] | [+0.0770, +0.4947] | [13, 14] | 2 | 768 | 0.9831 | a_gap_underpowered |
| Kvasir-Capsule K_lit | dinov2_vitb14 | mahalanobis_l2 | leaky | +0.7207 | [+0.5741, +0.8673] | [+0.5932, +0.8935] | [13, 14] | 2 | 768 | 0.9762 | a_gap_underpowered |
| Kvasir-Capsule K_lit | dinov2_vitb14 | mahalanobis_l2 | F2 | +0.4702 | [+0.2060, +0.7345] | [+0.3301, +0.7514] | [13, 14] | 2 | 768 | 0.9762 | a_gap_underpowered |
| Kvasir-Capsule K_lit | dinov2_vitb14 | mahalanobis_l2 | delta_fit | +0.2505 | [+0.1111, +0.3899] | [+0.1343, +0.3201] | [13, 14] | 2 | 768 | 0.9762 | a_gap_underpowered |
| Kvasir-Capsule K_lit | dinov2_vitb14 | mahalanobis_l2 | truth | +0.6029 | [+0.3732, +0.8326] | [+0.4378, +0.8482] | [13, 14] | 2 | 768 | 0.9762 | a_gap_underpowered |
| Kvasir-Capsule K_lit | dinov2_vitb14 | mahalanobis_l2 | a_gap | +0.1178 | [-0.0168, +0.2523] | [+0.0112, +0.2422] | [13, 14] | 2 | 768 | 0.9762 | a_gap_underpowered |
| Kvasir-Capsule K_lit | dinov2_vitb14 | mahalanobis_l2 | a_gap_cm | +0.1267 | [-0.0029, +0.2563] | [+0.0179, +0.2512] | [13, 14] | 2 | 768 | 0.9762 | a_gap_underpowered |
| Kvasir-Capsule K_lit | dinov2_vitb14 | knn_mean_cosine | leaky | +0.8035 | [+0.7470, +0.8601] | [+0.6574, +0.8835] | [13, 14] | 2 | 768 | 0.9762 | a_gap_underpowered |
| Kvasir-Capsule K_lit | dinov2_vitb14 | knn_mean_cosine | F2 | +0.5128 | [+0.4090, +0.6166] | [+0.3241, +0.6618] | [13, 14] | 2 | 768 | 0.9762 | a_gap_underpowered |
| Kvasir-Capsule K_lit | dinov2_vitb14 | knn_mean_cosine | delta_fit | +0.2907 | [+0.2023, +0.3792] | [+0.2131, +0.3658] | [13, 14] | 2 | 768 | 0.9762 | a_gap_underpowered |
| Kvasir-Capsule K_lit | dinov2_vitb14 | knn_mean_cosine | truth | +0.6424 | [+0.5224, +0.7624] | [+0.4254, +0.7817] | [13, 14] | 2 | 768 | 0.9762 | a_gap_underpowered |
| Kvasir-Capsule K_lit | dinov2_vitb14 | knn_mean_cosine | a_gap | +0.1611 | [+0.0598, +0.2625] | [+0.0739, +0.2730] | [13, 14] | 2 | 768 | 0.9762 | a_gap_underpowered |
| Kvasir-Capsule K_lit | dinov2_vitb14 | knn_mean_cosine | a_gap_cm | +0.1699 | [+0.0621, +0.2777] | [+0.0805, +0.2826] | [13, 14] | 2 | 768 | 0.9762 | a_gap_underpowered |
| Kvasir-Capsule K_lit | biomedclip | mahalanobis_l2 | leaky | +0.7064 | [+0.4979, +0.9149] | [+0.6317, +0.8970] | [13, 14] | 2 | 512 | 0.9236 | a_gap_underpowered |
| Kvasir-Capsule K_lit | biomedclip | mahalanobis_l2 | F2 | +0.5855 | [+0.2999, +0.8710] | [+0.5004, +0.8395] | [13, 14] | 2 | 512 | 0.9236 | a_gap_underpowered |
| Kvasir-Capsule K_lit | biomedclip | mahalanobis_l2 | delta_fit | +0.1209 | [+0.0361, +0.2058] | [+0.0547, +0.1474] | [13, 14] | 2 | 512 | 0.9236 | a_gap_underpowered |
| Kvasir-Capsule K_lit | biomedclip | mahalanobis_l2 | truth | +0.5244 | [+0.1809, +0.8679] | [+0.3838, +0.8228] | [13, 14] | 2 | 512 | 0.9236 | a_gap_underpowered |
| Kvasir-Capsule K_lit | biomedclip | mahalanobis_l2 | a_gap | +0.1820 | [+0.0075, +0.3566] | [+0.0517, +0.3070] | [13, 14] | 2 | 512 | 0.9236 | a_gap_underpowered |
| Kvasir-Capsule K_lit | biomedclip | mahalanobis_l2 | a_gap_cm | +0.1856 | [+0.0234, +0.3477] | [+0.0515, +0.3123] | [13, 14] | 2 | 512 | 0.9236 | a_gap_underpowered |
| Kvasir-Capsule K_lit | biomedclip | knn_mean_cosine | leaky | +0.7323 | [+0.4841, +0.9805] | [+0.6549, +0.9290] | [13, 14] | 2 | 512 | 0.9236 | below_chance, a_gap_underpowered |
| Kvasir-Capsule K_lit | biomedclip | knn_mean_cosine | F2 | +0.5108 | [+0.0876, +0.9339] | [+0.4028, +0.8466] | [13, 14] | 2 | 512 | 0.9236 | below_chance, a_gap_underpowered |
| Kvasir-Capsule K_lit | biomedclip | knn_mean_cosine | delta_fit | +0.2215 | [+0.0414, +0.4017] | [+0.0841, +0.2693] | [13, 14] | 2 | 512 | 0.9236 | below_chance, a_gap_underpowered |
| Kvasir-Capsule K_lit | biomedclip | knn_mean_cosine | truth | +0.4565 | [-0.0001, +0.9132] | [+0.3045, +0.8259] | [13, 14] | 2 | 512 | 0.9236 | below_chance, a_gap_underpowered |
| Kvasir-Capsule K_lit | biomedclip | knn_mean_cosine | a_gap | +0.2758 | [+0.0415, +0.5100] | [+0.0912, +0.4084] | [13, 14] | 2 | 512 | 0.9236 | below_chance, a_gap_underpowered |
| Kvasir-Capsule K_lit | biomedclip | knn_mean_cosine | a_gap_cm | +0.2815 | [+0.0520, +0.5110] | [+0.0909, +0.4152] | [13, 14] | 2 | 512 | 0.9236 | below_chance, a_gap_underpowered |
| Kvasir-Capsule K_seg | resnet50_s42 | mahalanobis_l2 | leaky | +0.5343 | [+0.2392, +0.8293] | [+0.3924, +0.7596] | [12, 17] | 2 | 2048 | 0.8073 | near_chance, a_gap_underpowered |
| Kvasir-Capsule K_seg | resnet50_s42 | mahalanobis_l2 | F2 | +0.3820 | [+0.0071, +0.7569] | [+0.2308, +0.6542] | [12, 17] | 2 | 2048 | 0.8073 | near_chance, a_gap_underpowered |
| Kvasir-Capsule K_seg | resnet50_s42 | mahalanobis_l2 | delta_fit | +0.1523 | [-0.0021, +0.3066] | [+0.0570, +0.2309] | [12, 17] | 2 | 2048 | 0.8073 | near_chance, a_gap_underpowered |
| Kvasir-Capsule K_seg | resnet50_s42 | mahalanobis_l2 | leaky_gap | +0.5416 | [+0.2544, +0.8288] | [+0.3994, +0.7610] | [12, 17] | 2 | 2048 | 0.8073 | near_chance, a_gap_underpowered |
| Kvasir-Capsule K_seg | resnet50_s42 | mahalanobis_l2 | delta_fit_gap | +0.1596 | [-0.0001, +0.3192] | [+0.0598, +0.2386] | [12, 17] | 2 | 2048 | 0.8073 | near_chance, a_gap_underpowered |
| Kvasir-Capsule K_seg | resnet50_s42 | mahalanobis_l2 | gap_effect | -0.0073 | [-0.0267, +0.0121] | [-0.0129, +0.0045] | [12, 17] | 2 | 2048 | 0.8073 | near_chance, a_gap_underpowered |
| Kvasir-Capsule K_seg | resnet50_s42 | mahalanobis_l2 | truth | +0.4942 | [+0.1221, +0.8664] | [+0.3742, +0.7651] | [12, 17] | 2 | 2048 | 0.8073 | near_chance, a_gap_underpowered |
| Kvasir-Capsule K_seg | resnet50_s42 | mahalanobis_l2 | a_gap | +0.0400 | [-0.1744, +0.2544] | [-0.1216, +0.1910] | [12, 17] | 2 | 2048 | 0.8073 | near_chance, a_gap_underpowered |
| Kvasir-Capsule K_seg | resnet50_s42 | mahalanobis_l2 | a_gap_cm | +0.0337 | [-0.1899, +0.2574] | [-0.1414, +0.2018] | [12, 17] | 2 | 2048 | 0.8073 | near_chance, a_gap_underpowered |
| Kvasir-Capsule K_seg | resnet50_s42 | knn_mean_cosine | leaky | +0.5764 | [+0.4272, +0.7256] | [+0.4423, +0.7169] | [12, 17] | 2 | 2048 | 0.8073 | a_gap_underpowered |
| Kvasir-Capsule K_seg | resnet50_s42 | knn_mean_cosine | F2 | +0.4611 | [+0.2490, +0.6732] | [+0.3264, +0.6513] | [12, 17] | 2 | 2048 | 0.8073 | a_gap_underpowered |
| Kvasir-Capsule K_seg | resnet50_s42 | knn_mean_cosine | delta_fit | +0.1153 | [-0.0246, +0.2552] | [+0.0321, +0.1711] | [12, 17] | 2 | 2048 | 0.8073 | a_gap_underpowered |
| Kvasir-Capsule K_seg | resnet50_s42 | knn_mean_cosine | leaky_gap | +0.5758 | [+0.4262, +0.7254] | [+0.4411, +0.7176] | [12, 17] | 2 | 2048 | 0.8073 | a_gap_underpowered |
| Kvasir-Capsule K_seg | resnet50_s42 | knn_mean_cosine | delta_fit_gap | +0.1147 | [-0.0252, +0.2546] | [+0.0352, +0.1670] | [12, 17] | 2 | 2048 | 0.8073 | a_gap_underpowered |
| Kvasir-Capsule K_seg | resnet50_s42 | knn_mean_cosine | gap_effect | +0.0006 | [-0.0031, +0.0043] | [-0.0033, +0.0047] | [12, 17] | 2 | 2048 | 0.8073 | a_gap_underpowered |
| Kvasir-Capsule K_seg | resnet50_s42 | knn_mean_cosine | truth | +0.5190 | [+0.3294, +0.7085] | [+0.4161, +0.6946] | [12, 17] | 2 | 2048 | 0.8073 | a_gap_underpowered |
| Kvasir-Capsule K_seg | resnet50_s42 | knn_mean_cosine | a_gap | +0.0574 | [-0.1616, +0.2765] | [-0.1304, +0.2165] | [12, 17] | 2 | 2048 | 0.8073 | a_gap_underpowered |
| Kvasir-Capsule K_seg | resnet50_s42 | knn_mean_cosine | a_gap_cm | +0.0430 | [-0.1850, +0.2710] | [-0.1674, +0.2184] | [12, 17] | 2 | 2048 | 0.8073 | a_gap_underpowered |
| Kvasir-Capsule K_seg | resnet50_s43 | mahalanobis_l2 | leaky | +0.5716 | [+0.1367, +1.0065] | [+0.4124, +0.9023] | [12, 17] | 2 | 2048 | 0.7911 | below_chance, a_gap_underpowered |
| Kvasir-Capsule K_seg | resnet50_s43 | mahalanobis_l2 | F2 | +0.4478 | [-0.0950, +0.9905] | [+0.2839, +0.8521] | [12, 17] | 2 | 2048 | 0.7911 | below_chance, a_gap_underpowered |
| Kvasir-Capsule K_seg | resnet50_s43 | mahalanobis_l2 | delta_fit | +0.1238 | [-0.0459, +0.2935] | [+0.0318, +0.1953] | [12, 17] | 2 | 2048 | 0.7911 | below_chance, a_gap_underpowered |
| Kvasir-Capsule K_seg | resnet50_s43 | mahalanobis_l2 | leaky_gap | +0.5704 | [+0.1333, +1.0074] | [+0.4088, +0.9024] | [12, 17] | 2 | 2048 | 0.7911 | below_chance, a_gap_underpowered |
| Kvasir-Capsule K_seg | resnet50_s43 | mahalanobis_l2 | delta_fit_gap | +0.1226 | [-0.0447, +0.2898] | [+0.0327, +0.1894] | [12, 17] | 2 | 2048 | 0.7911 | below_chance, a_gap_underpowered |
| Kvasir-Capsule K_seg | resnet50_s43 | mahalanobis_l2 | gap_effect | +0.0012 | [-0.0100, +0.0125] | [-0.0055, +0.0123] | [12, 17] | 2 | 2048 | 0.7911 | below_chance, a_gap_underpowered |
| Kvasir-Capsule K_seg | resnet50_s43 | mahalanobis_l2 | truth | +0.4520 | [-0.1129, +1.0170] | [+0.2440, +0.8873] | [12, 17] | 2 | 2048 | 0.7911 | below_chance, a_gap_underpowered |
| Kvasir-Capsule K_seg | resnet50_s43 | mahalanobis_l2 | a_gap | +0.1195 | [-0.1291, +0.3682] | [-0.0614, +0.3114] | [12, 17] | 2 | 2048 | 0.7911 | below_chance, a_gap_underpowered |
| Kvasir-Capsule K_seg | resnet50_s43 | mahalanobis_l2 | a_gap_cm | +0.1292 | [-0.0900, +0.3483] | [-0.0593, +0.3294] | [12, 17] | 2 | 2048 | 0.7911 | below_chance, a_gap_underpowered |
| Kvasir-Capsule K_seg | resnet50_s43 | knn_mean_cosine | leaky | +0.5128 | [-0.0028, +1.0285] | [+0.3733, +0.8561] | [12, 17] | 2 | 2048 | 0.7911 | below_chance, a_gap_underpowered |
| Kvasir-Capsule K_seg | resnet50_s43 | knn_mean_cosine | F2 | +0.4368 | [-0.1415, +1.0152] | [+0.2913, +0.8014] | [12, 17] | 2 | 2048 | 0.7911 | below_chance, a_gap_underpowered |
| Kvasir-Capsule K_seg | resnet50_s43 | knn_mean_cosine | delta_fit | +0.0760 | [-0.2439, +0.3959] | [+0.0123, +0.1553] | [12, 17] | 2 | 2048 | 0.7911 | below_chance, a_gap_underpowered |
| Kvasir-Capsule K_seg | resnet50_s43 | knn_mean_cosine | leaky_gap | +0.5090 | [-0.0133, +1.0313] | [+0.3665, +0.8566] | [12, 17] | 2 | 2048 | 0.7911 | below_chance, a_gap_underpowered |
| Kvasir-Capsule K_seg | resnet50_s43 | knn_mean_cosine | delta_fit_gap | +0.0721 | [-0.2474, +0.3916] | [+0.0102, +0.1497] | [12, 17] | 2 | 2048 | 0.7911 | below_chance, a_gap_underpowered |
| Kvasir-Capsule K_seg | resnet50_s43 | knn_mean_cosine | gap_effect | +0.0039 | [-0.0050, +0.0127] | [-0.0051, +0.0102] | [12, 17] | 2 | 2048 | 0.7911 | below_chance, a_gap_underpowered |
| Kvasir-Capsule K_seg | resnet50_s43 | knn_mean_cosine | truth | +0.4013 | [-0.1697, +0.9722] | [+0.2304, +0.8109] | [12, 17] | 2 | 2048 | 0.7911 | below_chance, a_gap_underpowered |
| Kvasir-Capsule K_seg | resnet50_s43 | knn_mean_cosine | a_gap | +0.1116 | [-0.0862, +0.3093] | [-0.0602, +0.2688] | [12, 17] | 2 | 2048 | 0.7911 | below_chance, a_gap_underpowered |
| Kvasir-Capsule K_seg | resnet50_s43 | knn_mean_cosine | a_gap_cm | +0.1124 | [-0.0637, +0.2886] | [-0.0632, +0.2806] | [12, 17] | 2 | 2048 | 0.7911 | below_chance, a_gap_underpowered |
| Kvasir-Capsule K_seg | convnext_tiny_s42 | mahalanobis_l2 | leaky | +0.4602 | [-0.1984, +1.1187] | [+0.2510, +0.9230] | [12, 17] | 2 | 768 | 0.7635 | near_chance, a_gap_underpowered |
| Kvasir-Capsule K_seg | convnext_tiny_s42 | mahalanobis_l2 | F2 | +0.4118 | [-0.2815, +1.1051] | [+0.2114, +0.9035] | [12, 17] | 2 | 768 | 0.7635 | near_chance, a_gap_underpowered |
| Kvasir-Capsule K_seg | convnext_tiny_s42 | mahalanobis_l2 | delta_fit | +0.0484 | [-0.0186, +0.1154] | [+0.0110, +0.0901] | [12, 17] | 2 | 768 | 0.7635 | near_chance, a_gap_underpowered |
| Kvasir-Capsule K_seg | convnext_tiny_s42 | mahalanobis_l2 | leaky_gap | +0.4617 | [-0.1917, +1.1151] | [+0.2528, +0.9206] | [12, 17] | 2 | 768 | 0.7635 | near_chance, a_gap_underpowered |
| Kvasir-Capsule K_seg | convnext_tiny_s42 | mahalanobis_l2 | delta_fit_gap | +0.0499 | [-0.0212, +0.1210] | [+0.0098, +0.0928] | [12, 17] | 2 | 768 | 0.7635 | near_chance, a_gap_underpowered |
| Kvasir-Capsule K_seg | convnext_tiny_s42 | mahalanobis_l2 | gap_effect | -0.0015 | [-0.0082, +0.0051] | [-0.0039, +0.0039] | [12, 17] | 2 | 768 | 0.7635 | near_chance, a_gap_underpowered |
| Kvasir-Capsule K_seg | convnext_tiny_s42 | mahalanobis_l2 | truth | +0.4885 | [-0.0883, +1.0653] | [+0.2667, +0.9054] | [12, 17] | 2 | 768 | 0.7635 | near_chance, a_gap_underpowered |
| Kvasir-Capsule K_seg | convnext_tiny_s42 | mahalanobis_l2 | a_gap | -0.0283 | [-0.2937, +0.2371] | [-0.2424, +0.2008] | [12, 17] | 2 | 768 | 0.7635 | near_chance, a_gap_underpowered |
| Kvasir-Capsule K_seg | convnext_tiny_s42 | mahalanobis_l2 | a_gap_cm | -0.0047 | [-0.2044, +0.1951] | [-0.2235, +0.2270] | [12, 17] | 2 | 768 | 0.7635 | near_chance, a_gap_underpowered |
| Kvasir-Capsule K_seg | convnext_tiny_s42 | knn_mean_cosine | leaky | +0.4747 | [-0.1742, +1.1236] | [+0.2714, +0.9203] | [12, 17] | 2 | 768 | 0.7635 | near_chance, a_gap_underpowered |
| Kvasir-Capsule K_seg | convnext_tiny_s42 | knn_mean_cosine | F2 | +0.3964 | [-0.3124, +1.1051] | [+0.2044, +0.8909] | [12, 17] | 2 | 768 | 0.7635 | near_chance, a_gap_underpowered |
| Kvasir-Capsule K_seg | convnext_tiny_s42 | knn_mean_cosine | delta_fit | +0.0784 | [-0.0249, +0.1816] | [+0.0218, +0.1336] | [12, 17] | 2 | 768 | 0.7635 | near_chance, a_gap_underpowered |
| Kvasir-Capsule K_seg | convnext_tiny_s42 | knn_mean_cosine | leaky_gap | +0.4730 | [-0.1760, +1.1219] | [+0.2697, +0.9190] | [12, 17] | 2 | 768 | 0.7635 | near_chance, a_gap_underpowered |
| Kvasir-Capsule K_seg | convnext_tiny_s42 | knn_mean_cosine | delta_fit_gap | +0.0766 | [-0.0268, +0.1800] | [+0.0205, +0.1317] | [12, 17] | 2 | 768 | 0.7635 | near_chance, a_gap_underpowered |
| Kvasir-Capsule K_seg | convnext_tiny_s42 | knn_mean_cosine | gap_effect | +0.0018 | [-0.0007, +0.0042] | [+0.0001, +0.0052] | [12, 17] | 2 | 768 | 0.7635 | near_chance, a_gap_underpowered |
| Kvasir-Capsule K_seg | convnext_tiny_s42 | knn_mean_cosine | truth | +0.4662 | [-0.1197, +1.0521] | [+0.2565, +0.8900] | [12, 17] | 2 | 768 | 0.7635 | near_chance, a_gap_underpowered |
| Kvasir-Capsule K_seg | convnext_tiny_s42 | knn_mean_cosine | a_gap | +0.0085 | [-0.2422, +0.2592] | [-0.1935, +0.2285] | [12, 17] | 2 | 768 | 0.7635 | near_chance, a_gap_underpowered |
| Kvasir-Capsule K_seg | convnext_tiny_s42 | knn_mean_cosine | a_gap_cm | +0.0322 | [-0.1587, +0.2232] | [-0.1766, +0.2533] | [12, 17] | 2 | 768 | 0.7635 | near_chance, a_gap_underpowered |
| Kvasir-Capsule K_seg | convnext_tiny_s43 | mahalanobis_l2 | leaky | +0.5181 | [+0.0124, +1.0237] | [+0.3369, +0.8893] | [12, 17] | 2 | 768 | 0.7940 | near_chance, a_gap_underpowered |
| Kvasir-Capsule K_seg | convnext_tiny_s43 | mahalanobis_l2 | F2 | +0.4675 | [-0.0721, +1.0071] | [+0.2877, +0.8693] | [12, 17] | 2 | 768 | 0.7940 | near_chance, a_gap_underpowered |
| Kvasir-Capsule K_seg | convnext_tiny_s43 | mahalanobis_l2 | delta_fit | +0.0506 | [-0.0119, +0.1130] | [+0.0136, +0.0860] | [12, 17] | 2 | 768 | 0.7940 | near_chance, a_gap_underpowered |
| Kvasir-Capsule K_seg | convnext_tiny_s43 | mahalanobis_l2 | leaky_gap | +0.5189 | [+0.0184, +1.0193] | [+0.3369, +0.8870] | [12, 17] | 2 | 768 | 0.7940 | near_chance, a_gap_underpowered |
| Kvasir-Capsule K_seg | convnext_tiny_s43 | mahalanobis_l2 | delta_fit_gap | +0.0513 | [-0.0147, +0.1173] | [+0.0117, +0.0870] | [12, 17] | 2 | 768 | 0.7940 | near_chance, a_gap_underpowered |
| Kvasir-Capsule K_seg | convnext_tiny_s43 | mahalanobis_l2 | gap_effect | -0.0008 | [-0.0079, +0.0063] | [-0.0034, +0.0055] | [12, 17] | 2 | 768 | 0.7940 | near_chance, a_gap_underpowered |
| Kvasir-Capsule K_seg | convnext_tiny_s43 | mahalanobis_l2 | truth | +0.5047 | [+0.0134, +0.9960] | [+0.2887, +0.8786] | [12, 17] | 2 | 768 | 0.7940 | near_chance, a_gap_underpowered |
| Kvasir-Capsule K_seg | convnext_tiny_s43 | mahalanobis_l2 | a_gap | +0.0134 | [-0.2383, +0.2651] | [-0.1998, +0.2329] | [12, 17] | 2 | 768 | 0.7940 | near_chance, a_gap_underpowered |
| Kvasir-Capsule K_seg | convnext_tiny_s43 | mahalanobis_l2 | a_gap_cm | +0.0442 | [-0.1501, +0.2385] | [-0.1669, +0.2625] | [12, 17] | 2 | 768 | 0.7940 | near_chance, a_gap_underpowered |
| Kvasir-Capsule K_seg | convnext_tiny_s43 | knn_mean_cosine | leaky | +0.5241 | [+0.0750, +0.9732] | [+0.3470, +0.8591] | [12, 17] | 2 | 768 | 0.7940 | near_chance, a_gap_underpowered |
| Kvasir-Capsule K_seg | convnext_tiny_s43 | knn_mean_cosine | F2 | +0.4555 | [-0.0635, +0.9745] | [+0.2891, +0.8415] | [12, 17] | 2 | 768 | 0.7940 | near_chance, a_gap_underpowered |
| Kvasir-Capsule K_seg | convnext_tiny_s43 | knn_mean_cosine | delta_fit | +0.0686 | [-0.0412, +0.1784] | [+0.0037, +0.1157] | [12, 17] | 2 | 768 | 0.7940 | near_chance, a_gap_underpowered |
| Kvasir-Capsule K_seg | convnext_tiny_s43 | knn_mean_cosine | leaky_gap | +0.5218 | [+0.0728, +0.9708] | [+0.3452, +0.8581] | [12, 17] | 2 | 768 | 0.7940 | near_chance, a_gap_underpowered |
| Kvasir-Capsule K_seg | convnext_tiny_s43 | knn_mean_cosine | delta_fit_gap | +0.0663 | [-0.0435, +0.1761] | [+0.0027, +0.1124] | [12, 17] | 2 | 768 | 0.7940 | near_chance, a_gap_underpowered |
| Kvasir-Capsule K_seg | convnext_tiny_s43 | knn_mean_cosine | gap_effect | +0.0023 | [-0.0005, +0.0051] | [+0.0003, +0.0064] | [12, 17] | 2 | 768 | 0.7940 | near_chance, a_gap_underpowered |
| Kvasir-Capsule K_seg | convnext_tiny_s43 | knn_mean_cosine | truth | +0.5029 | [+0.0601, +0.9458] | [+0.3004, +0.8441] | [12, 17] | 2 | 768 | 0.7940 | near_chance, a_gap_underpowered |
| Kvasir-Capsule K_seg | convnext_tiny_s43 | knn_mean_cosine | a_gap | +0.0211 | [-0.2280, +0.2703] | [-0.1890, +0.2408] | [12, 17] | 2 | 768 | 0.7940 | near_chance, a_gap_underpowered |
| Kvasir-Capsule K_seg | convnext_tiny_s43 | knn_mean_cosine | a_gap_cm | +0.0491 | [-0.1521, +0.2503] | [-0.1649, +0.2735] | [12, 17] | 2 | 768 | 0.7940 | near_chance, a_gap_underpowered |
| Kvasir-Capsule K_seg | dinov2_vitb14 | mahalanobis_l2 | leaky | +0.4956 | [+0.2285, +0.7628] | [+0.3325, +0.7661] | [12, 17] | 2 | 768 | 0.7643 | below_chance, a_gap_underpowered |
| Kvasir-Capsule K_seg | dinov2_vitb14 | mahalanobis_l2 | F2 | +0.4466 | [+0.1723, +0.7209] | [+0.2667, +0.7215] | [12, 17] | 2 | 768 | 0.7643 | below_chance, a_gap_underpowered |
| Kvasir-Capsule K_seg | dinov2_vitb14 | mahalanobis_l2 | delta_fit | +0.0491 | [-0.0075, +0.1056] | [+0.0015, +0.1023] | [12, 17] | 2 | 768 | 0.7643 | below_chance, a_gap_underpowered |
| Kvasir-Capsule K_seg | dinov2_vitb14 | mahalanobis_l2 | leaky_gap | +0.4943 | [+0.2284, +0.7602] | [+0.3312, +0.7640] | [12, 17] | 2 | 768 | 0.7643 | below_chance, a_gap_underpowered |
| Kvasir-Capsule K_seg | dinov2_vitb14 | mahalanobis_l2 | delta_fit_gap | +0.0477 | [-0.0083, +0.1037] | [+0.0003, +0.0997] | [12, 17] | 2 | 768 | 0.7643 | below_chance, a_gap_underpowered |
| Kvasir-Capsule K_seg | dinov2_vitb14 | mahalanobis_l2 | gap_effect | +0.0014 | [-0.0042, +0.0069] | [-0.0007, +0.0053] | [12, 17] | 2 | 768 | 0.7643 | below_chance, a_gap_underpowered |
| Kvasir-Capsule K_seg | dinov2_vitb14 | mahalanobis_l2 | truth | +0.5868 | [+0.3427, +0.8309] | [+0.4086, +0.8336] | [12, 17] | 2 | 768 | 0.7643 | below_chance, a_gap_underpowered |
| Kvasir-Capsule K_seg | dinov2_vitb14 | mahalanobis_l2 | a_gap | -0.0912 | [-0.2390, +0.0567] | [-0.2184, +0.0467] | [12, 17] | 2 | 768 | 0.7643 | below_chance, a_gap_underpowered |
| Kvasir-Capsule K_seg | dinov2_vitb14 | mahalanobis_l2 | a_gap_cm | -0.0620 | [-0.2221, +0.0980] | [-0.1885, +0.0829] | [12, 17] | 2 | 768 | 0.7643 | below_chance, a_gap_underpowered |
| Kvasir-Capsule K_seg | dinov2_vitb14 | knn_mean_cosine | leaky | +0.5312 | [+0.3937, +0.6686] | [+0.3156, +0.6969] | [12, 17] | 2 | 768 | 0.7643 | a_gap_underpowered |
| Kvasir-Capsule K_seg | dinov2_vitb14 | knn_mean_cosine | F2 | +0.4573 | [+0.2808, +0.6338] | [+0.2678, +0.6320] | [12, 17] | 2 | 768 | 0.7643 | a_gap_underpowered |
| Kvasir-Capsule K_seg | dinov2_vitb14 | knn_mean_cosine | delta_fit | +0.0738 | [-0.0009, +0.1486] | [+0.0190, +0.1181] | [12, 17] | 2 | 768 | 0.7643 | a_gap_underpowered |
| Kvasir-Capsule K_seg | dinov2_vitb14 | knn_mean_cosine | leaky_gap | +0.5286 | [+0.3908, +0.6664] | [+0.3133, +0.6952] | [12, 17] | 2 | 768 | 0.7643 | a_gap_underpowered |
| Kvasir-Capsule K_seg | dinov2_vitb14 | knn_mean_cosine | delta_fit_gap | +0.0713 | [-0.0035, +0.1461] | [+0.0167, +0.1156] | [12, 17] | 2 | 768 | 0.7643 | a_gap_underpowered |
| Kvasir-Capsule K_seg | dinov2_vitb14 | knn_mean_cosine | gap_effect | +0.0025 | [-0.0008, +0.0059] | [+0.0005, +0.0065] | [12, 17] | 2 | 768 | 0.7643 | a_gap_underpowered |
| Kvasir-Capsule K_seg | dinov2_vitb14 | knn_mean_cosine | truth | +0.6071 | [+0.4724, +0.7419] | [+0.4093, +0.7641] | [12, 17] | 2 | 768 | 0.7643 | a_gap_underpowered |
| Kvasir-Capsule K_seg | dinov2_vitb14 | knn_mean_cosine | a_gap | -0.0760 | [-0.1998, +0.0479] | [-0.1918, +0.0519] | [12, 17] | 2 | 768 | 0.7643 | a_gap_underpowered |
| Kvasir-Capsule K_seg | dinov2_vitb14 | knn_mean_cosine | a_gap_cm | -0.0506 | [-0.2035, +0.1022] | [-0.1682, +0.0840] | [12, 17] | 2 | 768 | 0.7643 | a_gap_underpowered |
| Kvasir-Capsule K_seg | biomedclip | mahalanobis_l2 | leaky | +0.5775 | [+0.2965, +0.8584] | [+0.4529, +0.8216] | [12, 17] | 2 | 512 | 0.7712 | a_gap_underpowered |
| Kvasir-Capsule K_seg | biomedclip | mahalanobis_l2 | F2 | +0.5352 | [+0.2382, +0.8322] | [+0.4068, +0.7885] | [12, 17] | 2 | 512 | 0.7712 | a_gap_underpowered |
| Kvasir-Capsule K_seg | biomedclip | mahalanobis_l2 | delta_fit | +0.0422 | [-0.0045, +0.0890] | [+0.0136, +0.0745] | [12, 17] | 2 | 512 | 0.7712 | a_gap_underpowered |
| Kvasir-Capsule K_seg | biomedclip | mahalanobis_l2 | leaky_gap | +0.5760 | [+0.2946, +0.8574] | [+0.4508, +0.8194] | [12, 17] | 2 | 512 | 0.7712 | a_gap_underpowered |
| Kvasir-Capsule K_seg | biomedclip | mahalanobis_l2 | delta_fit_gap | +0.0408 | [-0.0064, +0.0880] | [+0.0117, +0.0731] | [12, 17] | 2 | 512 | 0.7712 | a_gap_underpowered |
| Kvasir-Capsule K_seg | biomedclip | mahalanobis_l2 | gap_effect | +0.0015 | [-0.0018, +0.0047] | [+0.0002, +0.0037] | [12, 17] | 2 | 512 | 0.7712 | a_gap_underpowered |
| Kvasir-Capsule K_seg | biomedclip | mahalanobis_l2 | truth | +0.5258 | [+0.1943, +0.8574] | [+0.3723, +0.8159] | [12, 17] | 2 | 512 | 0.7712 | a_gap_underpowered |
| Kvasir-Capsule K_seg | biomedclip | mahalanobis_l2 | a_gap | +0.0516 | [-0.1231, +0.2263] | [-0.0791, +0.2074] | [12, 17] | 2 | 512 | 0.7712 | a_gap_underpowered |
| Kvasir-Capsule K_seg | biomedclip | mahalanobis_l2 | a_gap_cm | +0.0720 | [-0.0913, +0.2353] | [-0.0680, +0.2307] | [12, 17] | 2 | 512 | 0.7712 | a_gap_underpowered |
| Kvasir-Capsule K_seg | biomedclip | knn_mean_cosine | leaky | +0.5020 | [+0.0746, +0.9295] | [+0.3524, +0.8384] | [12, 17] | 2 | 512 | 0.7712 | near_chance, a_gap_underpowered |
| Kvasir-Capsule K_seg | biomedclip | knn_mean_cosine | F2 | +0.4546 | [+0.0050, +0.9041] | [+0.3060, +0.8071] | [12, 17] | 2 | 512 | 0.7712 | near_chance, a_gap_underpowered |
| Kvasir-Capsule K_seg | biomedclip | knn_mean_cosine | delta_fit | +0.0475 | [-0.0014, +0.0963] | [+0.0196, +0.0765] | [12, 17] | 2 | 512 | 0.7712 | near_chance, a_gap_underpowered |
| Kvasir-Capsule K_seg | biomedclip | knn_mean_cosine | leaky_gap | +0.5002 | [+0.0719, +0.9284] | [+0.3503, +0.8359] | [12, 17] | 2 | 512 | 0.7712 | near_chance, a_gap_underpowered |
| Kvasir-Capsule K_seg | biomedclip | knn_mean_cosine | delta_fit_gap | +0.0456 | [-0.0027, +0.0939] | [+0.0187, +0.0741] | [12, 17] | 2 | 512 | 0.7712 | near_chance, a_gap_underpowered |
| Kvasir-Capsule K_seg | biomedclip | knn_mean_cosine | gap_effect | +0.0019 | [-0.0011, +0.0049] | [+0.0001, +0.0053] | [12, 17] | 2 | 512 | 0.7712 | near_chance, a_gap_underpowered |
| Kvasir-Capsule K_seg | biomedclip | knn_mean_cosine | truth | +0.4525 | [-0.0073, +0.9122] | [+0.3018, +0.8199] | [12, 17] | 2 | 512 | 0.7712 | near_chance, a_gap_underpowered |
| Kvasir-Capsule K_seg | biomedclip | knn_mean_cosine | a_gap | +0.0496 | [-0.1278, +0.2269] | [-0.0925, +0.2097] | [12, 17] | 2 | 512 | 0.7712 | near_chance, a_gap_underpowered |
| Kvasir-Capsule K_seg | biomedclip | knn_mean_cosine | a_gap_cm | +0.0703 | [-0.0769, +0.2175] | [-0.0737, +0.2324] | [12, 17] | 2 | 512 | 0.7712 | near_chance, a_gap_underpowered |
| brain MRI | resnet50_s42 | mahalanobis_l2 | leaky | +0.9255 | [+0.8854, +0.9657] | [+0.8885, +0.9539] | [60, 60] | 2 | 2048 | 0.9913 | - |
| brain MRI | resnet50_s42 | mahalanobis_l2 | F2 | +0.8917 | [+0.8406, +0.9428] | [+0.8476, +0.9272] | [60, 60] | 2 | 2048 | 0.9913 | - |
| brain MRI | resnet50_s42 | mahalanobis_l2 | delta_fit | +0.0338 | [+0.0170, +0.0507] | [+0.0233, +0.0451] | [60, 60] | 2 | 2048 | 0.9913 | - |
| brain MRI | resnet50_s42 | mahalanobis_l2 | truth | +0.8934 | [+0.8249, +0.9619] | [+0.8346, +0.9420] | [60, 60] | 2 | 2048 | 0.9913 | - |
| brain MRI | resnet50_s42 | mahalanobis_l2 | a_gap | +0.0322 | [-0.0254, +0.0897] | [-0.0164, +0.0865] | [60, 60] | 2 | 2048 | 0.9913 | - |
| brain MRI | resnet50_s42 | mahalanobis_l2 | a_gap_cm | +0.0322 | [-0.0250, +0.0895] | [-0.0161, +0.0862] | [60, 60] | 2 | 2048 | 0.9913 | - |
| brain MRI | resnet50_s42 | knn_mean_cosine | leaky | +0.9049 | [+0.8561, +0.9537] | [+0.8603, +0.9409] | [60, 60] | 2 | 2048 | 0.9913 | - |
| brain MRI | resnet50_s42 | knn_mean_cosine | F2 | +0.8995 | [+0.8512, +0.9478] | [+0.8552, +0.9358] | [60, 60] | 2 | 2048 | 0.9913 | - |
| brain MRI | resnet50_s42 | knn_mean_cosine | delta_fit | +0.0054 | [-0.0009, +0.0117] | [+0.0016, +0.0091] | [60, 60] | 2 | 2048 | 0.9913 | - |
| brain MRI | resnet50_s42 | knn_mean_cosine | truth | +0.8965 | [+0.8318, +0.9611] | [+0.8384, +0.9437] | [60, 60] | 2 | 2048 | 0.9913 | - |
| brain MRI | resnet50_s42 | knn_mean_cosine | a_gap | +0.0085 | [-0.0426, +0.0595] | [-0.0387, +0.0588] | [60, 60] | 2 | 2048 | 0.9913 | - |
| brain MRI | resnet50_s42 | knn_mean_cosine | a_gap_cm | +0.0087 | [-0.0410, +0.0584] | [-0.0380, +0.0590] | [60, 60] | 2 | 2048 | 0.9913 | - |
| brain MRI | resnet50_s43 | mahalanobis_l2 | leaky | +0.9296 | [+0.8923, +0.9668] | [+0.8953, +0.9561] | [60, 60] | 2 | 2048 | 0.9796 | - |
| brain MRI | resnet50_s43 | mahalanobis_l2 | F2 | +0.8787 | [+0.8237, +0.9337] | [+0.8341, +0.9156] | [60, 60] | 2 | 2048 | 0.9796 | - |
| brain MRI | resnet50_s43 | mahalanobis_l2 | delta_fit | +0.0509 | [+0.0250, +0.0767] | [+0.0362, +0.0682] | [60, 60] | 2 | 2048 | 0.9796 | - |
| brain MRI | resnet50_s43 | mahalanobis_l2 | truth | +0.8839 | [+0.8163, +0.9515] | [+0.8220, +0.9369] | [60, 60] | 2 | 2048 | 0.9796 | - |
| brain MRI | resnet50_s43 | mahalanobis_l2 | a_gap | +0.0457 | [-0.0145, +0.1058] | [-0.0088, +0.1075] | [60, 60] | 2 | 2048 | 0.9796 | - |
| brain MRI | resnet50_s43 | mahalanobis_l2 | a_gap_cm | +0.0453 | [-0.0164, +0.1070] | [-0.0091, +0.1069] | [60, 60] | 2 | 2048 | 0.9796 | - |
| brain MRI | resnet50_s43 | knn_mean_cosine | leaky | +0.8995 | [+0.8508, +0.9482] | [+0.8549, +0.9353] | [60, 60] | 2 | 2048 | 0.9796 | - |
| brain MRI | resnet50_s43 | knn_mean_cosine | F2 | +0.8905 | [+0.8404, +0.9405] | [+0.8441, +0.9269] | [60, 60] | 2 | 2048 | 0.9796 | - |
| brain MRI | resnet50_s43 | knn_mean_cosine | delta_fit | +0.0090 | [+0.0007, +0.0173] | [+0.0050, +0.0135] | [60, 60] | 2 | 2048 | 0.9796 | - |
| brain MRI | resnet50_s43 | knn_mean_cosine | truth | +0.8843 | [+0.8172, +0.9515] | [+0.8217, +0.9382] | [60, 60] | 2 | 2048 | 0.9796 | - |
| brain MRI | resnet50_s43 | knn_mean_cosine | a_gap | +0.0152 | [-0.0442, +0.0745] | [-0.0424, +0.0766] | [60, 60] | 2 | 2048 | 0.9796 | - |
| brain MRI | resnet50_s43 | knn_mean_cosine | a_gap_cm | +0.0151 | [-0.0444, +0.0747] | [-0.0421, +0.0761] | [60, 60] | 2 | 2048 | 0.9796 | - |
| brain MRI | convnext_tiny_s42 | mahalanobis_l2 | leaky | +0.8856 | [+0.8309, +0.9403] | [+0.8341, +0.9264] | [60, 60] | 2 | 768 | 0.9738 | - |
| brain MRI | convnext_tiny_s42 | mahalanobis_l2 | F2 | +0.8392 | [+0.7750, +0.9034] | [+0.7776, +0.8891] | [60, 60] | 2 | 768 | 0.9738 | - |
| brain MRI | convnext_tiny_s42 | mahalanobis_l2 | delta_fit | +0.0464 | [+0.0284, +0.0645] | [+0.0315, +0.0623] | [60, 60] | 2 | 768 | 0.9738 | - |
| brain MRI | convnext_tiny_s42 | mahalanobis_l2 | truth | +0.8466 | [+0.7576, +0.9356] | [+0.7671, +0.9100] | [60, 60] | 2 | 768 | 0.9738 | - |
| brain MRI | convnext_tiny_s42 | mahalanobis_l2 | a_gap | +0.0390 | [-0.0425, +0.1205] | [-0.0268, +0.1096] | [60, 60] | 2 | 768 | 0.9738 | - |
| brain MRI | convnext_tiny_s42 | mahalanobis_l2 | a_gap_cm | +0.0407 | [-0.0296, +0.1110] | [-0.0249, +0.1115] | [60, 60] | 2 | 768 | 0.9738 | - |
| brain MRI | convnext_tiny_s42 | knn_mean_cosine | leaky | +0.8611 | [+0.8018, +0.9204] | [+0.8040, +0.9088] | [60, 60] | 2 | 768 | 0.9738 | - |
| brain MRI | convnext_tiny_s42 | knn_mean_cosine | F2 | +0.8548 | [+0.7946, +0.9150] | [+0.7964, +0.9038] | [60, 60] | 2 | 768 | 0.9738 | - |
| brain MRI | convnext_tiny_s42 | knn_mean_cosine | delta_fit | +0.0063 | [+0.0012, +0.0115] | [+0.0032, +0.0096] | [60, 60] | 2 | 768 | 0.9738 | - |
| brain MRI | convnext_tiny_s42 | knn_mean_cosine | truth | +0.8394 | [+0.7533, +0.9255] | [+0.7574, +0.9047] | [60, 60] | 2 | 768 | 0.9738 | - |
| brain MRI | convnext_tiny_s42 | knn_mean_cosine | a_gap | +0.0217 | [-0.0529, +0.0963] | [-0.0429, +0.0897] | [60, 60] | 2 | 768 | 0.9738 | - |
| brain MRI | convnext_tiny_s42 | knn_mean_cosine | a_gap_cm | +0.0239 | [-0.0353, +0.0831] | [-0.0404, +0.0918] | [60, 60] | 2 | 768 | 0.9738 | - |
| brain MRI | convnext_tiny_s43 | mahalanobis_l2 | leaky | +0.9045 | [+0.8665, +0.9425] | [+0.8696, +0.9331] | [60, 60] | 2 | 768 | 0.9971 | - |
| brain MRI | convnext_tiny_s43 | mahalanobis_l2 | F2 | +0.8649 | [+0.8152, +0.9146] | [+0.8227, +0.9019] | [60, 60] | 2 | 768 | 0.9971 | - |
| brain MRI | convnext_tiny_s43 | mahalanobis_l2 | delta_fit | +0.0396 | [+0.0193, +0.0600] | [+0.0273, +0.0541] | [60, 60] | 2 | 768 | 0.9971 | - |
| brain MRI | convnext_tiny_s43 | mahalanobis_l2 | truth | +0.8640 | [+0.7898, +0.9382] | [+0.8015, +0.9162] | [60, 60] | 2 | 768 | 0.9971 | - |
| brain MRI | convnext_tiny_s43 | mahalanobis_l2 | a_gap | +0.0405 | [-0.0234, +0.1044] | [-0.0104, +0.0989] | [60, 60] | 2 | 768 | 0.9971 | - |
| brain MRI | convnext_tiny_s43 | mahalanobis_l2 | a_gap_cm | +0.0394 | [-0.0275, +0.1063] | [-0.0116, +0.0976] | [60, 60] | 2 | 768 | 0.9971 | - |
| brain MRI | convnext_tiny_s43 | knn_mean_cosine | leaky | +0.9021 | [+0.8631, +0.9410] | [+0.8647, +0.9310] | [60, 60] | 2 | 768 | 0.9971 | - |
| brain MRI | convnext_tiny_s43 | knn_mean_cosine | F2 | +0.8942 | [+0.8539, +0.9345] | [+0.8559, +0.9246] | [60, 60] | 2 | 768 | 0.9971 | - |
| brain MRI | convnext_tiny_s43 | knn_mean_cosine | delta_fit | +0.0079 | [-0.0004, +0.0162] | [+0.0038, +0.0123] | [60, 60] | 2 | 768 | 0.9971 | - |
| brain MRI | convnext_tiny_s43 | knn_mean_cosine | truth | +0.8829 | [+0.8170, +0.9487] | [+0.8273, +0.9306] | [60, 60] | 2 | 768 | 0.9971 | - |
| brain MRI | convnext_tiny_s43 | knn_mean_cosine | a_gap | +0.0192 | [-0.0385, +0.0769] | [-0.0286, +0.0723] | [60, 60] | 2 | 768 | 0.9971 | - |
| brain MRI | convnext_tiny_s43 | knn_mean_cosine | a_gap_cm | +0.0186 | [-0.0418, +0.0790] | [-0.0289, +0.0716] | [60, 60] | 2 | 768 | 0.9971 | - |
| brain MRI | dinov2_vitb14 | mahalanobis_l2 | leaky | +0.7340 | [+0.6764, +0.7916] | [+0.6814, +0.7814] | [60, 60] | 2 | 768 | 0.9913 | - |
| brain MRI | dinov2_vitb14 | mahalanobis_l2 | F2 | +0.6461 | [+0.5809, +0.7114] | [+0.5883, +0.6984] | [60, 60] | 2 | 768 | 0.9913 | - |
| brain MRI | dinov2_vitb14 | mahalanobis_l2 | delta_fit | +0.0879 | [+0.0694, +0.1064] | [+0.0732, +0.1037] | [60, 60] | 2 | 768 | 0.9913 | - |
| brain MRI | dinov2_vitb14 | mahalanobis_l2 | truth | +0.6872 | [+0.6225, +0.7518] | [+0.6286, +0.7421] | [60, 60] | 2 | 768 | 0.9913 | - |
| brain MRI | dinov2_vitb14 | mahalanobis_l2 | a_gap | +0.0468 | [-0.0143, +0.1079] | [-0.0187, +0.1114] | [60, 60] | 2 | 768 | 0.9913 | - |
| brain MRI | dinov2_vitb14 | mahalanobis_l2 | a_gap_cm | +0.0502 | [+0.0057, +0.0947] | [-0.0156, +0.1145] | [60, 60] | 2 | 768 | 0.9913 | - |
| brain MRI | dinov2_vitb14 | knn_mean_cosine | leaky | +0.6613 | [+0.6001, +0.7225] | [+0.6034, +0.7152] | [60, 60] | 2 | 768 | 0.9913 | - |
| brain MRI | dinov2_vitb14 | knn_mean_cosine | F2 | +0.6222 | [+0.5610, +0.6833] | [+0.5649, +0.6776] | [60, 60] | 2 | 768 | 0.9913 | - |
| brain MRI | dinov2_vitb14 | knn_mean_cosine | delta_fit | +0.0391 | [+0.0258, +0.0525] | [+0.0307, +0.0475] | [60, 60] | 2 | 768 | 0.9913 | - |
| brain MRI | dinov2_vitb14 | knn_mean_cosine | truth | +0.6699 | [+0.5883, +0.7515] | [+0.5945, +0.7427] | [60, 60] | 2 | 768 | 0.9913 | - |
| brain MRI | dinov2_vitb14 | knn_mean_cosine | a_gap | -0.0086 | [-0.0902, +0.0731] | [-0.0934, +0.0733] | [60, 60] | 2 | 768 | 0.9913 | - |
| brain MRI | dinov2_vitb14 | knn_mean_cosine | a_gap_cm | -0.0035 | [-0.0443, +0.0373] | [-0.0884, +0.0783] | [60, 60] | 2 | 768 | 0.9913 | - |
| brain MRI | biomedclip | mahalanobis_l2 | leaky | +0.8252 | [+0.7678, +0.8826] | [+0.7677, +0.8760] | [60, 60] | 2 | 512 | 0.9883 | - |
| brain MRI | biomedclip | mahalanobis_l2 | F2 | +0.7796 | [+0.7147, +0.8445] | [+0.7144, +0.8376] | [60, 60] | 2 | 512 | 0.9883 | - |
| brain MRI | biomedclip | mahalanobis_l2 | delta_fit | +0.0456 | [+0.0277, +0.0636] | [+0.0306, +0.0613] | [60, 60] | 2 | 512 | 0.9883 | - |
| brain MRI | biomedclip | mahalanobis_l2 | truth | +0.7950 | [+0.7156, +0.8745] | [+0.7213, +0.8651] | [60, 60] | 2 | 512 | 0.9883 | - |
| brain MRI | biomedclip | mahalanobis_l2 | a_gap | +0.0302 | [-0.0544, +0.1147] | [-0.0479, +0.1106] | [60, 60] | 2 | 512 | 0.9883 | - |
| brain MRI | biomedclip | mahalanobis_l2 | a_gap_cm | +0.0335 | [-0.0275, +0.0944] | [-0.0442, +0.1132] | [60, 60] | 2 | 512 | 0.9883 | - |
| brain MRI | biomedclip | knn_mean_cosine | leaky | +0.8286 | [+0.7683, +0.8889] | [+0.7649, +0.8819] | [60, 60] | 2 | 512 | 0.9883 | - |
| brain MRI | biomedclip | knn_mean_cosine | F2 | +0.8010 | [+0.7365, +0.8655] | [+0.7366, +0.8582] | [60, 60] | 2 | 512 | 0.9883 | - |
| brain MRI | biomedclip | knn_mean_cosine | delta_fit | +0.0275 | [+0.0092, +0.0458] | [+0.0154, +0.0403] | [60, 60] | 2 | 512 | 0.9883 | - |
| brain MRI | biomedclip | knn_mean_cosine | truth | +0.8124 | [+0.7351, +0.8898] | [+0.7391, +0.8803] | [60, 60] | 2 | 512 | 0.9883 | - |
| brain MRI | biomedclip | knn_mean_cosine | a_gap | +0.0161 | [-0.0588, +0.0910] | [-0.0556, +0.0902] | [60, 60] | 2 | 512 | 0.9883 | - |
| brain MRI | biomedclip | knn_mean_cosine | a_gap_cm | +0.0196 | [-0.0304, +0.0697] | [-0.0523, +0.0939] | [60, 60] | 2 | 512 | 0.9883 | - |

## CXR (P2-a, descriptive)

| backbone | scorer | Delta_fit (Shenzhen) |
|---|---|---|
| resnet50_s42 | mahalanobis_l2 | +0.0139 |
| resnet50_s42 | knn_mean_cosine | +0.0179 |
| resnet50_s43 | mahalanobis_l2 | +0.0159 |
| resnet50_s43 | knn_mean_cosine | +0.0194 |
| convnext_tiny_s42 | mahalanobis_l2 | +0.0297 |
| convnext_tiny_s42 | knn_mean_cosine | +0.0315 |
| convnext_tiny_s43 | mahalanobis_l2 | +0.0227 |
| convnext_tiny_s43 | knn_mean_cosine | +0.0260 |
| dinov2_vitb14 | mahalanobis_l2 | +0.0155 |
| dinov2_vitb14 | knn_mean_cosine | +0.0154 |
| rad_dino | mahalanobis_l2 | +0.0214 |
| rad_dino | knn_mean_cosine | +0.0893 |

## Caveats

1. Brain MRI has no slice index, so it has no adjacent-slice gap variant; adjacent-slice similarity stays in the brain estimate and can only raise it (residual asymmetry against the ordering prediction).
2. K_seg is a mechanistic control, not a protocol from the literature; K_lit is the literature-protocol effect size (frame-level random split, Li et al. 2023; El-Ghany et al. 2024), reported without a gap.
3. A_gap is descriptive; Kvasir-Capsule has 7 unseen videos (A_gap underpowered by construction).
4. Seen images whose group has no fit image are dropped and counted: Kvasir-Capsule K_lit 0; Kvasir-Capsule K_seg 157; brain MRI 4.
5. Backbone aggregation over seeds (mean; robust = every seed's CI > 0; ordering bounds = min / max over seeds) is not specified in the precommit and was fixed before the cells were computed.
6. Package check: leaky and F2 match crossfit_auroc within 1e-9 plus one pair per near-tied seen / OOD score pair (the P2-b technical fix, applied from the start here).
7. Process deviation (commit history): the first REPORT commit (2eb9c17) was a STOP report written by the auto-finalize watcher, whose generator call failed on a log path containing '/' before the generator ran; the chain had completed. Replaced by the generated REPORT on the same data (c3df6ed) in 9648ca0; no data or analysis changed. The SLURM report job (64827) was cancelled while queued; the same generator ran on the login node.
