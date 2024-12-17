/*
 Example showing how to use the Animation class with the cart-pole system
*/

#include "animation.hpp"
#include <cmath>

int main() {
    // Create animation with custom configuration
    Animation::AnimationConfig config;
    config.width = 800;
    config.height = 600;
    config.frame_skip = 2;  // Save every other frame
    config.frame_delay = 10;  // Faster animation
    Animation animation(config);

    // Set up cart-pole parameters
    double cart_width = 0.3;
    double cart_height = 0.2;
    double pole_width = 0.05;
    double pole_length = 1.0;

    // Simulate cart-pole motion
    int num_frames = 200;
    double dt = 0.05;
    double theta = M_PI/3;  // Initial angle (60 degrees)
    double omega = 0.0;     // Initial angular velocity
    double x = 0.0;         // Initial cart position

    for (int i = 0; i < num_frames; ++i) {
        animation.newFrame();

        // Update simulation
        double g = 9.81;
        omega += -3*g/(2*pole_length) * sin(theta) * dt;
        theta += omega * dt;
        x = sin(i * 0.05) * 0.5;  // Simple sinusoidal cart motion

        // Draw cart
        std::vector<double> cart_x = {
            x - cart_width/2, x + cart_width/2,
            x + cart_width/2, x - cart_width/2,
            x - cart_width/2
        };
        std::vector<double> cart_y = {
            -cart_height/2, -cart_height/2,
            cart_height/2, cart_height/2,
            -cart_height/2
        };
        plt::plot(cart_x, cart_y, "k-");

        // Draw pole
        double pole_end_x = x + pole_length * sin(theta);
        double pole_end_y = pole_length * cos(theta);
        std::vector<double> pole_x = {x, pole_end_x};
        std::vector<double> pole_y = {0, pole_end_y};
        plt::plot(pole_x, pole_y, "b-");

        // Draw pole bob
        std::vector<double> circle_x, circle_y;
        for (int j = 0; j <= 20; ++j) {
            double t = 2 * M_PI * j / 20;
            circle_x.push_back(pole_end_x + pole_width * cos(t));
            circle_y.push_back(pole_end_y + pole_width * sin(t));
        }
        plt::plot(circle_x, circle_y, "b-");

        // Set fixed axis limits
        plt::xlim(x - 2, x + 2);
        plt::ylim(-2, 2);
        
        // Save the frame
        animation.saveFrame(i);
    }

    // Create the final GIF
    animation.createGif("cartpole_animation.gif");

    return 0;
}