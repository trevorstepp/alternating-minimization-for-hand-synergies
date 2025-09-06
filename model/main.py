from model.methods.alternating_min import AlternatingMinModel
from model.methods.two_stage import TwoStage

if __name__ == '__main__':
    T = 10
    t_s = 5
    m = 6
    n = 4
    K_j = T - t_s + 1
    G = 2
    lambda1 = 0.05
    alpha = 0.05
    altMin = AlternatingMinModel(T, t_s, m, n, K_j, G, lambda1=lambda1, alpha=alpha)
    print("**Alternating Minimization Model Run**")
    print(f"{altMin.true_C}")
    altMin.solve()

    print("\n\n")

    twoStage = TwoStage(T, t_s, m, n, K_j, G, lambda1=lambda1, alpha=alpha)
    print("**Two Stage Model Run**")
    #twoStage.solve()