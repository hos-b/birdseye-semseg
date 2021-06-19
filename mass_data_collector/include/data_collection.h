#ifndef __MASS_DATA_COLLECTION_H__
#define __MASS_DATA_COLLECTION_H__

#include <string>
#include <mutex>
#include <yaml-cpp/yaml.h>
#include <ros/package.h>
#include <unordered_map>
#include <algorithm>
#include <utility>

struct NoiseSetting
{
    NoiseSetting(){};
    NoiseSetting(const YAML::Node& noise_node) {
        auto agent_yaw_node = noise_node["agent-yaw"];
        agent_yaw_enable = agent_yaw_node["enable"].as<bool>();
        agent_yaw_chance = agent_yaw_node["chance"].as<float>();
        agent_yaw_mean = agent_yaw_node["mean"].as<float>();
        agent_yaw_std = agent_yaw_node["std"].as<float>();
        auto front_rgb_pitch_node = noise_node["front-rgb-pitch"];
        front_rgb_pitch_enable = front_rgb_pitch_node["enable"].as<bool>();
        front_rgb_pitch_chance = front_rgb_pitch_node["chance"].as<float>();
        front_rgb_pitch_mean = front_rgb_pitch_node["mean"].as<float>();
        front_rgb_pitch_std = front_rgb_pitch_node["std"].as<float>();
    }
    bool agent_yaw_enable;
    float agent_yaw_chance;
    float agent_yaw_mean;
    float agent_yaw_std;
    bool front_rgb_pitch_enable;
    float front_rgb_pitch_chance;
    float front_rgb_pitch_mean;
    float front_rgb_pitch_std;
};

struct CollectionConfig
{
    std::string dataset_path;
    std::string dataset_name;
    bool append;
    std::vector<unsigned int> towns;
    std::vector<std::string> weathers;
    std::vector<unsigned int> town_batch_counts;
    std::vector<unsigned int> cumulative_batch_counts;
    std::vector<float> batch_size_distribution;
    unsigned int maximum_cars;
    unsigned int minimum_cars;
    unsigned int max_batch_count;
    unsigned int hdf5_chunk_size;
    unsigned long random_seed;
    NoiseSetting noise_setting;
    // randomized batch sizes with enforced distribution
    std::vector<unsigned int> all_batch_sizes;

    static const CollectionConfig& GetConfig() {
        static CollectionConfig conf;
        static std::once_flag once;
        std::call_once(once, []() {
            std::set<std::string> allowed_weathers = {
                "Default",
                "ClearNoon",
                "CloudyNoon",
                "WetNoon",
                "WetCloudyNoon",
                "MidRainyNoon",
                "HardRainNoon",
                "SoftRainNoon",
                "ClearSunset",
                "CloudySunset",
                "WetSunset",
                "WetCloudySunset",
                "MidRainSunset",
                "HardRainSunset",
                "SoftRainSunset"
            };
            std::set<int> allowed_towns = {
                1, 2, 3, 4, 5
            };
            std::string yaml_path = ros::package::getPath("mass_data_collector") +
								    "/param/data_collection.yaml";
		    YAML::Node base = YAML::LoadFile(yaml_path);
		    YAML::Node dataset = base["dataset"];
		    YAML::Node collection = base["collection"];
		    conf.dataset_path = dataset["path"].as<std::string>();
            conf.dataset_name = dataset["name"].as<std::string>();
            conf.append = dataset["append"].as<bool>();
            // reading noise parameters
            conf.noise_setting = NoiseSetting(base["noise"]);
            // reading town list
            auto yaml_towns = dataset["towns"];
            if (!yaml_towns.IsSequence()) {
                std::cout << "towns should be a list" << std::endl;
                std::exit(EXIT_FAILURE);
            }
            for (const auto &yaml_town : yaml_towns) {
                if (!yaml_town.IsScalar()) {
                    std::cout << "expected scalars in towns" << std::endl;
                    std::exit(EXIT_FAILURE);
                }
                int town_id = yaml_town.as<int>();
                if (allowed_towns.find(town_id) == allowed_towns.end()) {
                    std::cout << town_id << " is not a valid town id" << std::endl;
                    std::exit(EXIT_FAILURE);
                }
                conf.towns.emplace_back(town_id);
            }
            // reading towns' weathers
            auto yaml_weathers = dataset["weathers"];
            if (!yaml_weathers.IsSequence()) {
                std::cout << "weathers should be a list" << std::endl;
                std::exit(EXIT_FAILURE);
            }
            if (yaml_weathers.size() != yaml_towns.size()) {
                std::cout << "weathers list length does not match number of towns" << std::endl;
                std::exit(EXIT_FAILURE);
            }
            for (const auto &yaml_weather : yaml_weathers) {
                std::string town_weather = yaml_weather.as<std::string>();
                if (allowed_weathers.find(town_weather) == allowed_weathers.end()) {
                    std::cout << town_weather << " is not a valid weather condition" << std::endl;
                    std::exit(EXIT_FAILURE);
                }
                conf.weathers.emplace_back(town_weather);
            }
            // min/max agent count in the scenes
            conf.maximum_cars = collection["maximum-cars"].as<unsigned int>();
            // distribution of batche sizes
            auto yaml_distribution = collection["batch-size-distribution"];
            if (!yaml_distribution.IsSequence()) {
                std::cout << "batch-size-distribution should be a list" << std::endl;
                std::exit(EXIT_FAILURE);
            }
            if (yaml_distribution.size() != conf.maximum_cars) {
                std::cout << "batch-size-distribution does not match the number of agents" << std::endl;
                std::exit(EXIT_FAILURE);
            }
            float total_prob = 0.0f;
            bool min_found = false;
            for (unsigned int i = 0; i < yaml_distribution.size(); ++i) {
                if (!yaml_distribution[i].IsScalar()) {
                    std::cout << "expected scalars in batch-size-distribution" << std::endl;
                    std::exit(EXIT_FAILURE);
                }
                conf.batch_size_distribution.emplace_back(yaml_distribution[i].as<float>());
                if (!min_found && conf.batch_size_distribution.back() > 0) {
                    min_found = true;
                    conf.minimum_cars = i + 1;
                }
                total_prob += conf.batch_size_distribution.back();
            }
            if (total_prob != 1.0f) {
                    std::cout << "probabilities in batch-size-distribution do not sum up to 1.0" << std::endl;
                    std::exit(EXIT_FAILURE);
            }
            // max sample count, batch count in each town
            conf.max_batch_count = collection["maximum-batch-count"].as<unsigned int>();
            auto yaml_batch_counts = collection["town-batch-counts"];
            if (!yaml_batch_counts.IsSequence()) {
                std::cout << "town-batch-counts should be a list" << std::endl;
                std::exit(EXIT_FAILURE);
            }
            if (yaml_batch_counts.size() != yaml_towns.size()) {
                std::cout << "town-batch-counts does not match towns list" << std::endl;
                std::exit(EXIT_FAILURE);
            }
            // calculate commulative batch count for town switching
            size_t commulative_batch_count = 0;
            for (const auto &batch_count : yaml_batch_counts) {
                if (!batch_count.IsScalar()) {
                    std::cout << "expected scalars in town-batch-counts" << std::endl;
                    std::exit(EXIT_FAILURE);
                }
                conf.town_batch_counts.emplace_back(batch_count.as<size_t>());
                commulative_batch_count += conf.town_batch_counts.back();
                conf.cumulative_batch_counts.emplace_back(commulative_batch_count);
            }
            // misc. settings
            conf.hdf5_chunk_size = collection["hdf5-chunk-size"].as<unsigned int>();
            conf.random_seed = collection["random-seed"].as<unsigned long>();
            // calculate "random" batch sizes for all towns while enforcing distribution
            conf.all_batch_sizes.reserve(conf.cumulative_batch_counts.back());
            std::mt19937 random_gen(conf.random_seed);
            // filling batch sizes for all towns
            for (unsigned int town_index = 0; town_index < conf.towns.size(); ++town_index) {
                std::vector<unsigned int> town_batch_sizes;
                // filling batch sizes for a single town based on given distribution
                for (unsigned int agent_count = 1; agent_count <= conf.maximum_cars; ++agent_count) {
                    town_batch_sizes.insert(town_batch_sizes.end(),
                                            conf.town_batch_counts[town_index] * 
                                            conf.batch_size_distribution[agent_count - 1],
                                            agent_count);
                }
                // shuffling the order
                std::shuffle(town_batch_sizes.begin(), town_batch_sizes.end(), random_gen);
                // filling the missing batch sizes [due to removed floating point]
                std::uniform_int_distribution<unsigned int> distrib(0, town_batch_sizes.size() - 1);
                int missing_batches = conf.town_batch_counts[town_index] - town_batch_sizes.size();
                for (int j = 0; j < missing_batches; ++j) {
                    town_batch_sizes.emplace_back(town_batch_sizes[distrib(random_gen)]);
                }
                conf.all_batch_sizes.insert(conf.all_batch_sizes.end(),
                                            town_batch_sizes.begin(),
                                            town_batch_sizes.end());
            }
        });
        return conf;
    }
    /* return a string containing the name of all towns */
    std::string towns_string() const {
        if (towns.size() == 0) {
            return "";
        }
        std::string str = std::to_string(towns[0]);
        for (size_t i = 1; i < towns.size(); ++i) {
            str += ", " + std::to_string(towns[i]);
        }
        return str;
    }
    /* returns whether it's time to change towns given the index */
    std::tuple<bool, int, std::string> GetNewTown(size_t batch_index) const {
        if (batch_index == 0) {
            return std::make_tuple(true, towns[0], weathers[0]);
        }
        for (size_t i = 1; i < cumulative_batch_counts.size(); ++i) {
            if (batch_index == cumulative_batch_counts[i - 1]) {
                return std::make_tuple(true, towns[i], weathers[i]);
            }
        }
        return std::make_tuple(false, -1, "null");
    }
    /* returns the batch size given the index */
    unsigned int GetBatchSize(size_t batch_index) const {
        return all_batch_sizes[batch_index];
    }
};

struct CollectionStats
{
    CollectionStats(unsigned int max_agent_count) {
        for (unsigned int i = 1; i <= max_agent_count; ++i) {
            batch_histogram[i] = 0;
        }
    }
    void AddNewBatch(unsigned int batch_size) {
        batch_histogram[batch_size] = batch_histogram[batch_size] + 1;
    }
    std::vector<unsigned int> AsVector() {
        std::vector<unsigned int> ret(batch_histogram.size());
        for (auto& [_, value] : batch_histogram) {
            ret.insert(ret.begin(), value);
        }
        return ret;
    }
    void Print() {
        for (auto& [key, value] : batch_histogram) {
            std::cout << key << ": " << value << std::endl;
        }
    }
    std::unordered_map<unsigned int, unsigned int> batch_histogram;
};

#endif