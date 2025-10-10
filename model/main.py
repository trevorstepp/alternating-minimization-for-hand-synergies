from model.methods.alternating_min import AlternatingMinModel
from model.methods.two_stage import TwoStage

from model.utils.read_mat import load_natural_grasps
from dotenv import load_dotenv

load_dotenv()

if __name__ == '__main__':

    subject = "subj1"
    filename = "Natural_Test_Data.mat"

    n, T, G, V = load_natural_grasps(subject, filename)
    t_s = 39
    m = 10
    K_j = T - t_s + 1
    lambda1 = 0.25  # combination between group lasso and lasso (0=group lasso, 1=lasso)
    alpha = 0.00012  # overall regularization strength
    altMin = AlternatingMinModel(T, t_s, m, n, K_j, G, V, lambda1=lambda1, alpha=alpha)
    print("**Alternating Minimization Model Run**")
    #print(altMin.S.shape())
    altMin.solve()
    altMin.save_active_synergies(tol=1e-6)

    print("\n\n")

    #twoStage = TwoStage(T, t_s, m, n, K_j, G, lambda1=lambda1, alpha=alpha)
    #print("**Two Stage Model Run**")
    #twoStage.solve()