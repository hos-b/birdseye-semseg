#ifndef __MASS_DATA_COLLECTION_H__
#define __MASS_DATA_COLLECTION_H__

#include <string>
#include <mutex>
#include <yaml-cpp/yaml.h>
#include <ros/package.h>
#include <unordered_map>

struct CollectionConfig
{
    std::string dataset_path;
    std::string dataset_name;
    bool append;
    std::vector<size_t> towns;
    std::vector<size_t> town_batch_counts;
    std::vector<size_t> cumulative_batch_counts;
    unsigned int minimum_cars;
    unsigned int maximum_cars;
    unsigned int total_batch_count;
    unsigned int batch_delay_ms;
    unsigned int hdf5_chunk_size;
    unsigned long random_seed;
    float deadlock_multiplier;

    static const CollectionConfig& GetConfig() {
        static CollectionConfig conf;
        static std::once_flag once;
        std::call_once(once, []() {
            std::string yaml_path = ros::package::getPath("mass_data_collector") +
								    "/param/data_collection.yaml";
		    YAML::Node base = YAML::LoadFile(yaml_path);
		    YAML::Node dataset = base["dataset"];
		    YAML::Node collection = base["collection"];
		    conf.dataset_path = dataset["path"].as<std::string>();
            conf.dataset_name = dataset["name"].as<std::string>();
            conf.append = dataset["append"].as<bool>();
            auto yaml_towns = dataset["towns"];
            if (yaml_towns.IsSequence()) {
                for (const auto &yaml_town : yaml_towns) {
                    if (yaml_town.IsScalar()) {
                        conf.towns.emplace_back(yaml_town.as<size_t>());
                    }
                }
            }

            conf.minimum_cars = collection["minimum_cars"].as<unsigned int>();
            conf.maximum_cars = collection["maximum_cars"].as<unsigned int>();
            auto yaml_batch_counts = collection["town_batch_counts"];
            if (yaml_batch_counts.IsSequence()) {
                for (const auto &batch_count : yaml_batch_counts) {
                    if (batch_count.IsScalar()) {
                        conf.town_batch_counts.emplace_back(batch_count.as<size_t>());
                        conf.total_batch_count += conf.town_batch_counts.back();
                        conf.cumulative_batch_counts.emplace_back(conf.total_batch_count);
                    }
                }
            }
            conf.batch_delay_ms = collection["batch_delay_ms"].as<unsigned int>();
            conf.hdf5_chunk_size = collection["hdf5_chunk_size"].as<unsigned int>();
            conf.random_seed = collection["random_seed"].as<unsigned long>();
            conf.deadlock_multiplier = collection["deadlock_multiplier"].as<float>();
        });
        return conf;
    }
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
};

struct CollectionStats
{
    CollectionStats(unsigned int min_agent_count, unsigned int max_agent_count) {
        for (unsigned int i = min_agent_count; i <= max_agent_count; ++i) {
            batch_histogram[i] = 0;
        }
    }
    void AddNewBatch(unsigned int batch_size) {
        batch_histogram[batch_size] = batch_histogram[batch_size] + 1;
    }
    std::vector<unsigned int> AsVector() {
        std::vector<unsigned int> ret;
        for (auto& [_, value] : batch_histogram) {
            ret.emplace_back(value);
        }
        return ret;
    }
    std::unordered_map<unsigned int, unsigned int> batch_histogram;
};

#endif