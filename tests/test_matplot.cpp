#include <cmath>
#include <matplot/matplot.h>
#include <gtest/gtest.h>

TEST(MatplotTest, BasicPlot) {
    using namespace matplot;
    auto [X, Y] = meshgrid(linspace(-5, +5, 40), linspace(-5, +5, 40));
    auto Z = transform(X, Y, [](double x, double y) {
        return 10 * 2 + pow(x, 2) - 10 * cos(2 * pi * x) + pow(y, 2) -
               10 * cos(2 * pi * y);
    });
    surf(X, Y, Z);
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
