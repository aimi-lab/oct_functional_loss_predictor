from enum import Enum
from functools import total_ordering

@total_ordering
class GlaucomaStage(Enum):
    """Glaucoma stage according to R. P. Mills et al., “Categorizing the Stage of Glaucoma From Pre-Diagnosis 
    to End-Stage Disease,” American Journal of Ophthalmology, vol. 141, no. 1, pp. 24–30, Jan. 2006, doi: 10.1016/j.ajo.2005.07.044.

    """
    NORMAL = 0   # originally called "ocular hypertension / earliest glaucoma"
    EARLY = 1
    MODERATE = 2
    ADVANCED = 3
    SEVERE = 4
    END_STAGE = 5

    def __str__(self):
        return f"Stage {self.value}: {self.name}"
    
    def __repr__(self):
        return self.name
    
    def __lt__(self, other):
        if not isinstance(other, GlaucomaStage):
            return NotImplemented
        return self.value < other.value
    
    def __eq__(self, other):
        if not isinstance(other, GlaucomaStage):
            return NotImplemented
        return self.value == other.value
    
    def __hash__(self):
        return hash(self.value)

    @staticmethod
    def from_mean_deviation(mean_deviation: float, device: str):
        """Get the glaucoma stage from the mean deviation value.
            Depends on the device used to measure the visual field.
        """
        device = device.lower()
        if device == 'octopus':
            return GlaucomaStage.from_mean_deviation_octopus(mean_deviation)
        elif device == 'humphrey':
            return GlaucomaStage.from_mean_deviation_humphrey(mean_deviation)
        else:
            raise ValueError(f"Unknown device: {device}")

    @staticmethod
    def from_mean_deviation_octopus(mean_deviation: float):
        if mean_deviation <= -0.8:
            return GlaucomaStage.NORMAL
        elif mean_deviation <= 4.4:
            return GlaucomaStage.EARLY
        elif mean_deviation <= 9.5:
            return GlaucomaStage.MODERATE
        elif mean_deviation <= 15.3:
            return GlaucomaStage.ADVANCED
        elif mean_deviation <= 23.1:
            return GlaucomaStage.SEVERE
        else:
            return GlaucomaStage.END_STAGE
        
    @staticmethod
    def from_mean_deviation_humphrey(mean_deviation: float):
        if mean_deviation > 0.0:
            return GlaucomaStage.NORMAL
        elif mean_deviation > -5.0:
            return GlaucomaStage.EARLY
        elif mean_deviation > -12.0:
            return GlaucomaStage.MODERATE
        elif mean_deviation > -20.0:
            return GlaucomaStage.ADVANCED
        else:
            return GlaucomaStage.SEVERE
        # No end stage for Humphrey visual fields