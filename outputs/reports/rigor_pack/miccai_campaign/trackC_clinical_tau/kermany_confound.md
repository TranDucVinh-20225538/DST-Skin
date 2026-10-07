# Kermany Track C confound check (post-hoc, descriptive)

Class order CNV / DME / NORMAL. test_unseen = v2 test, patients absent from v2 train; unseen_extra = v3 supplement (patients absent from v2 train).

## Class histograms

```
         set   n  class_hist
   test_seen 664 242/200/222
 test_unseen  86     8/50/28
unseen_extra 750 250/250/250
```

## Accuracy per set (seed 42, std arm)

```
set                 test_seen  test_unseen  unseen_extra
arch                                                    
convnext_tiny           0.998        1.000         0.976
densenet121             0.994        1.000         0.979
effb3                   0.994        1.000         0.989
efficientnet_v2_s       0.997        1.000         0.991
mobilenet_v3_large      0.994        0.988         0.987
regnet_y_3_2gf          0.995        1.000         0.987
resnet18                0.997        1.000         0.968
resnet50                0.994        0.988         0.973
```

## Median over archs: realized TPR at τ (95% on test_seen)

```
            tpr_at_tau                          tpr_at_tau_class_matched                         
set          test_seen test_unseen unseen_extra                test_seen test_unseen unseen_extra
score                                                                                            
ELogitNorm       0.949       0.913        0.663                    0.949       0.913        0.669
Energy           0.949       0.878        0.697                    0.949       0.878        0.703
MSP              0.949       0.872        0.697                    0.949       0.872        0.702
Mahalanobis      0.949       0.890        0.392                    0.949       0.890        0.395
ReAct            0.949       0.884        0.696                    0.949       0.884        0.702
ViM              0.949       0.901        0.784                    0.949       0.901        0.786
kNN              0.949       0.878        0.495                    0.949       0.878        0.497
```
