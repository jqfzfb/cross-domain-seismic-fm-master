from . import modeling
from .modeling.criterion import SetMaskFormerCriterion, SetSegmentationCriterion, SetRegressionCriterion
from .modeling.matcher import HungarianMatcher

def make_loss_function(
    weight_dict, 
    num_classes, 
    weight=0.1,
    eos_coef = 0.1,
    mode = 'match',
):
    loss_keys = sorted(weight_dict.keys())
    
    if mode == 'match':
        matcher = HungarianMatcher(
            cost_class=weight_dict['ce'],
            cost_mask=weight_dict['focal'],
            cost_dice=weight_dict['dice'],
        )
        
        if loss_keys == ['ce', 'dice', 'focal', 'iou']:
            criterion = SetMaskFormerCriterion(
                num_classes,
                matcher=matcher,
                eos_coef=eos_coef,
                losses=['labels', 'masks', 'ious'],
            )    
        elif loss_keys == ['ce', 'dice', 'focal']:
            criterion = SetMaskFormerCriterion(
                num_classes,
                matcher=matcher,
                eos_coef=eos_coef,
                losses=['labels', 'masks'],
            )      
            
    elif mode in ['segment']:  
        if loss_keys == ['dice', 'focal']:
            criterion = SetSegmentationCriterion(
                losses=['masks'],
            )  
            
    elif mode in ['regress', 'inverse']:  
        if loss_keys == ['mse']:
            criterion = SetRegressionCriterion(
                losses=['mse'],
            ) 
        elif loss_keys == ['ssim']:
            criterion = SetRegressionCriterion(
                losses=['ssim'],
            ) 
    else:
        assert 0, f"Error: Unkown {mode}!"
    return criterion


