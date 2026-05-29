"""
Test I was doing to see if starting alternating minimization with learned synergies from 
another subject (rather than random initialization) leads to better results for this subject.
"""

from model.utils.read_files import load_natural_grasps
from model.utils.read_files import verify_saved_synergies
from model.utils.save_files import save_active_repeats
from model.methods.alternating_min import AlternatingMinModel
from dotenv import load_dotenv

load_dotenv()

if __name__ == '__main__':

    subject = "subj8"
    filename = "Natural_Test_Data.mat"

    synergies = verify_saved_synergies(subject="subj2", filename="subj2_synergies.mat")

    n, T, G, V = load_natural_grasps(subject, filename)
    n, t_s, m = synergies.shape
    print(f"n: {n}, t_s: {t_s}, m: {m}")
    K_j = T - t_s + 1
    lambda1 = 0.25  # combination between group lasso and lasso (0=group lasso, 1=lasso)
    alpha = 0.000085  # overall regularization strength
    altMin = AlternatingMinModel(T, t_s, m, n, K_j, G, subject, V, seed=1, lambda1=lambda1, alpha=alpha)
    altMin.s_list = [synergies[:, :, j] for j in range(altMin.m)]
    print([altMin.s_list[j].shape for j in range(altMin.m)])
    altMin.S = altMin.build_S(altMin.s_list, altMin.m)
    altMin.solve()
    save_active_repeats(altMin.S, subject, tol=1e-4)
