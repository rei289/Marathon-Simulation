// this is a test script
#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <iomanip>

struct Runner {
    std::string name;
    double speed;           
    double baseEndurance;   // Original endurance
    double currentEndurance; // Current endurance (degrades over time)
    double distanceCovered;
    bool hasFinished;
    double timeElapsed;     // Track time for this runner

    Runner(std::string n, double s, double e)
        : name(n), speed(s), baseEndurance(e), currentEndurance(e), 
          distanceCovered(0), hasFinished(false), timeElapsed(0) {}
};

class MarathonSimulation {
private:
    std::vector<Runner> runners;
    const double MARATHON_DISTANCE = 42195.0; // meters
    std::mt19937 rng;

public:
    MarathonSimulation() : rng(std::random_device{}()) {}

    void addRunner(const std::string& name, double speed, double endurance) {
        runners.emplace_back(name, speed, endurance);
    }

    void simulateStep(double timeStep = 1.0) {
        std::uniform_real_distribution<double> dist(0.8, 1.2);
        
        for (auto& runner : runners) {
            if (!runner.hasFinished) {
                runner.timeElapsed += timeStep;
                
                // Endurance degrades over time (fatigue)
                double fatigueRate = (1.0 - runner.baseEndurance) * 0.0001; // Adjust as needed
                runner.currentEndurance = std::max(0.3, runner.baseEndurance - (fatigueRate * runner.timeElapsed));
                
                double variation = dist(rng);
                double effectiveSpeed = runner.speed * runner.currentEndurance * variation;
                
                runner.distanceCovered += effectiveSpeed * timeStep;
                
                if (runner.distanceCovered >= MARATHON_DISTANCE) {
                    runner.distanceCovered = MARATHON_DISTANCE;
                    runner.hasFinished = true;
                }
            }
        }
    }

    void displayStatus(double timeElapsed) {
        int hours = static_cast<int>(timeElapsed / 3600);
        int minutes = static_cast<int>((timeElapsed - hours * 3600) / 60);
        
        std::cout << "Time: " << hours << "h " << minutes << "m\n";
        std::cout << std::string(60, '-') << '\n';
        
        for (const auto& runner : runners) {
            double percentage = (runner.distanceCovered / MARATHON_DISTANCE) * 100;
            double kmCovered = runner.distanceCovered / 1000.0;
            
            std::cout << std::left << std::setw(10) << runner.name 
                      << std::fixed << std::setprecision(2) << std::setw(8) << kmCovered << "km "
                      << std::setprecision(1) << std::setw(6) << percentage << "%";
            
            if (runner.hasFinished) {
                int finishHours = static_cast<int>(runner.timeElapsed / 3600);
                int finishMinutes = static_cast<int>((runner.timeElapsed - finishHours * 3600) / 60);
                std::cout << " - FINISHED! (" << finishHours << "h " << finishMinutes << "m)";
            }
            std::cout << '\n';
        }
        std::cout << '\n';
    }

    void runSimulation() {
        double timeElapsed = 0;
        const double MAX_TIME = 8 * 3600; // 8 hours in seconds
        const double UPDATE_INTERVAL = 600; // Update every 10 minutes
        const double TIME_STEP = 1.0; // 1 second steps

        while (timeElapsed < MAX_TIME) {
            simulateStep(TIME_STEP);
            timeElapsed += TIME_STEP;
            
            // Display status every 10 minutes
            if (static_cast<int>(timeElapsed) % static_cast<int>(UPDATE_INTERVAL) == 0) {
                displayStatus(timeElapsed);
            }

            // Check if all runners have finished
            bool allFinished = true;
            for (const auto& runner : runners) {
                if (!runner.hasFinished) {
                    allFinished = false;
                    break;
                }
            }
            
            if (allFinished) {
                displayStatus(timeElapsed);
                std::cout << "All runners have finished the marathon!\n";
                break;
            }
        }

        if (timeElapsed >= MAX_TIME) {
            std::cout << "Race time limit reached (8 hours)\n";
        }
    }
};

int main() {
    MarathonSimulation marathon;

    // Add runners with different characteristics
    marathon.addRunner("Eliud", 6.0, 0.95);   // Fast and high endurance
    marathon.addRunner("Sara", 5.8, 0.90);     // Fast but lower endurance
    marathon.addRunner("John", 4.5, 0.85);     // Average runner
    marathon.addRunner("Emma", 4.0, 0.99);     // Slower but excellent endurance
    marathon.addRunner("Mike", 3.8, 0.75);     // Beginner

    std::cout << "MARATHON SIMULATION STARTED!\n";
    std::cout << "Distance: 42.195 km\n\n";

    marathon.runSimulation();

    return 0;
}