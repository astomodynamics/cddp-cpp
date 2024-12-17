/*
 Copyright 2024 Tomo Sasaki

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
*/

#ifndef ANIMATION_HPP
#define ANIMATION_HPP

#include <string>
#include <vector>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <cstdlib>
#include "matplotlibcpp.hpp"

namespace plt = matplotlibcpp;
namespace fs = std::filesystem;

class Animation {
public:
    struct AnimationConfig {
        int width;               // Figure width in pixels
        int height;             // Figure height in pixels
        int frame_skip;         // Save every nth frame
        int frame_delay;        // Delay between frames in 1/100ths of a second
        std::string temp_dir;   // Directory for temporary frame files
        std::string output_dir; // Directory for output GIFs
        bool cleanup_frames;    // Whether to delete frame files after creating GIF

        AnimationConfig() :
            width(800),
            height(600),
            frame_skip(1),
            frame_delay(20),
            temp_dir("../results/frames"),
            output_dir("../results/animations"),
            cleanup_frames(true)
        {}
    };

    // Constructor
    explicit Animation(const AnimationConfig& config = AnimationConfig()) 
        : config_(config) {
        // Create directories if they don't exist
        fs::create_directories(config_.temp_dir);
        fs::create_directories(config_.output_dir);
        
        // Initialize figure
        plt::figure_size(config_.width, config_.height);
    }

    // Start a new frame
    void newFrame() {
        plt::clf();  // Clear current figure
    }

    // Save current frame
    void saveFrame(const int frame_number) {
        if (frame_number % config_.frame_skip == 0) {
            std::string filename = config_.temp_dir + "/frame_" + 
                                 std::to_string(frame_number/config_.frame_skip) + ".png";
            plt::save(filename);
        }
    }

    // Create GIF from saved frames
    bool createGif(const std::string& output_filename) {
        std::string command = "convert -delay " + std::to_string(config_.frame_delay) + " " +
                             config_.temp_dir + "/frame_*.png " +
                             config_.output_dir + "/" + output_filename;
        
        int result = std::system(command.c_str());
        
        if (result != 0) {
            std::cerr << "Failed to create GIF. Is ImageMagick installed?" << std::endl;
            return false;
        }

        // Clean up temporary frame files if requested
        if (config_.cleanup_frames) {
            cleanupFrames();
        }

        return true;
    }

    // Clean up temporary frame files
    void cleanupFrames() {
        for (const auto& entry : fs::directory_iterator(config_.temp_dir)) {
            fs::remove(entry.path());
        }
    }

    // Setter for configuration
    void setConfig(const AnimationConfig& config) {
        config_ = config;
        // Update figure size if changed
        plt::figure_size(config_.width, config_.height);
    }

    // Getter for configuration
    const AnimationConfig& getConfig() const { 
        return config_; 
    }

private:
    AnimationConfig config_;
};

#endif // ANIMATION_HPP