import numpy as np

from abc import ABC, abstractmethod


class FeatureGroup(ABC):
    def __init__(self, group_number: int, group_name: str):
        self.group_number = group_number
        self.group_name = group_name
        self.pca = None
        
    def make_copies(self, features_i: np.ndarray, 
                          features_k: np.ndarray, 
                          features_g: np.ndarray, 
                          features_r: np.ndarray):
        return (features_i.copy(), 
                features_k.copy(), 
                features_g.copy(),
                features_r.copy())
            
    @abstractmethod
    def __call__(self, features_i: np.ndarray, 
                       features_k: np.ndarray, 
                       features_g: np.ndarray, 
                       features_r: np.ndarray) -> np.ndarray:
        pass
            
#------------------------------------------------------------
#------------------------------------------------------------
#------------------------------------------------------------

class FeatureGroup1(FeatureGroup):

    INIT_QFC = 15
    INIT_BD  = 24

    def __init__(self):
        super().__init__(1, 'I')
    
    def __call__(self, features_i: np.ndarray, 
                       features_k: np.ndarray, 
                       features_g: np.ndarray, 
                       features_r: np.ndarray) -> np.ndarray:
        features_i = features_i.copy()
        return features_i[self.INIT_QFC:self.INIT_BD].reshape(1, -1)

#------------------------------------------------------------
#------------------------------------------------------------
#------------------------------------------------------------

class FeatureGroup2(FeatureGroup):

    def __init__(self):
        super().__init__(2, 'R')

    def __call__(self, features_i: np.ndarray, 
                       features_k: np.ndarray, 
                       features_g: np.ndarray, 
                       features_r: np.ndarray) -> np.ndarray:
        features_r = features_r.copy()
        return features_r.reshape(1, -1)

#------------------------------------------------------------
#------------------------------------------------------------
#------------------------------------------------------------

class FeatureGroup3(FeatureGroup):

    def __init__(self):
        super().__init__(3, 'I+')

    def __call__(self, features_i: np.ndarray, 
                       features_k: np.ndarray, 
                       features_g: np.ndarray, 
                       features_r: np.ndarray) -> np.ndarray:
        features_i = features_i.copy()
        return np.concatenate([features_i], axis=0).reshape(1, -1)
    
#------------------------------------------------------------
#------------------------------------------------------------
#------------------------------------------------------------
    
class FeatureGroup4(FeatureGroup):

    def __init__(self):
        super().__init__(4, 'K')
    
    def __call__(self, features_i: np.ndarray, 
                       features_k: np.ndarray, 
                       features_g: np.ndarray, 
                       features_r: np.ndarray) -> np.ndarray:
        features_k = features_k.copy()
        return np.concatenate([features_k], axis=0).reshape(1, -1)

#------------------------------------------------------------
#------------------------------------------------------------
#------------------------------------------------------------

class FeatureGroup6(FeatureGroup):
    
    def __init__(self):
        super().__init__(6, 'I+_K')
    
    def __call__(self, features_i: np.ndarray, 
                       features_k: np.ndarray, 
                       features_g: np.ndarray, 
                       features_r: np.ndarray) -> np.ndarray:
        features_i, features_k = features_i.copy(), features_k.copy()
        return np.concatenate([features_i, features_k], axis=0).reshape(1, -1)

#------------------------------------------------------------
#------------------------------------------------------------
#------------------------------------------------------------

class FeatureGroup7(FeatureGroup):
    
    def __init__(self):
        super().__init__(7, 'I+_G')
    
    
    def __call__(self, features_i: np.ndarray, 
                       features_k: np.ndarray, 
                       features_g: np.ndarray, 
                       features_r: np.ndarray) -> np.ndarray:
        features_i, features_g = features_i.copy(), features_g.copy()
        return np.concatenate([features_g, features_i], axis=0).reshape(1, -1)

#------------------------------------------------------------
#------------------------------------------------------------
#------------------------------------------------------------

class FeatureGroup10(FeatureGroup):

    def __init__(self):
        super().__init__(10, 'K_G')

    def __call__(self, features_i: np.ndarray, 
                       features_k: np.ndarray, 
                       features_g: np.ndarray, 
                       features_r: np.ndarray) -> np.ndarray:
        features_k, features_g = features_k.copy(), features_g.copy()
        return np.concatenate([features_g, features_k], axis=0).reshape(1, -1)

#------------------------------------------------------------
#------------------------------------------------------------
#------------------------------------------------------------

class FullGroup(FeatureGroup):

    def __init__(self):
        super().__init__(11, 'FG')

    def __call__(self, features_i: np.ndarray, 
                       features_k: np.ndarray, 
                       features_g: np.ndarray, 
                       features_r: np.ndarray) -> np.ndarray:
        features_i, features_k, features_g, features_r = \
            self.make_copies(features_i, features_k, features_g, features_r)
        return np.concatenate([features_g, features_i, features_g, features_k], axis=0).reshape(1, -1)

#------------------------------------------------------------
#------------------------------------------------------------
#------------------------------------------------------------

artifact_x_feature_selection = {
    'nobrain':          FeatureGroup2, 
    'folding':          FeatureGroup4, 
    'motion':           FeatureGroup4, 
    'gibbs':            FeatureGroup3, 
    'bdp':              FeatureGroup6, 
    'smoothing':        FeatureGroup1, 
    'zipper':           FeatureGroup1, 
    'noise':            FeatureGroup7,
    'bdl':              FullGroup, 
    'susceptibility':   FullGroup, 
    'lineshift':        FullGroup,
    'biasfield':        FeatureGroup10,
    'emptyline':        FullGroup
}

def get_feature_group(artifact: str) -> FeatureGroup:
    return artifact_x_feature_selection[artifact]()