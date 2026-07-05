import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        m = np.max(z)
        r =[]
        for i in z:
            r.append(np.exp(i - m))
        de = sum(r)
        for i in range(len(r)):
            r[i] = np.round(r[i]/de,4)
        return np.array(r)