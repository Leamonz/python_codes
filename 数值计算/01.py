# a = 0.0
# b = 2.0
# h = 1.0
# eps = 1e-9
#
#
# def f(x):
#     return x ** 3 - x - 1
#
#
# def bin(l, r):
#     while abs(l - r) > eps:
#         mid = (l + r) / 2
#         if f(mid) * f(l) < 0:
#             r = mid
#         elif f(mid) * f(r) < 0:
#             l = mid
#     return (l + r) / 2
#
#
# def scan(l, r, step):
#     x1 = l
#     x2 = x1 + step
#     while True:
#         if x2 > r:
#             step -= 0.1
#             x1 = l
#             x2 = x1 + step
#             continue
#         if f(x1) * f(x2) > 0:
#             x1 = x2
#             x2 = x1 + step
#         else:
#             break
#     return x1, x2
#
#
# # def g(x):
# #     return (1.0 + x) / (x ** 2)
# #
# #
# # def iter():
# #     xk = g(0.1)
# #     xk_1 = g(xk)
# #     while abs(xk_1 - xk) > eps:
# #         xk = g(xk_1)
# #         xk_1 = g(xk)
# #     return xk_1
#
#
# if __name__ == "__main__":
#     lend, rend = scan(a, b, h)
#     pred_x1 = bin(lend, rend)
#     print(f(pred_x1))
#     # pred_x2 = iter()
#     # print(f(pred_x2))
