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
    unsigned int minimum_cars;
    unsigned int maximum_cars;
    unsigned int max_batch_count;
    unsigned int batch_delay_ms;
    unsigned long random_seed;
    static CollectionConfig& GetConfig() {
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
            conf.minimum_cars = collection["minimum_cars"].as<unsigned int>();
            conf.maximum_cars = collection["maximum_cars"].as<unsigned int>();
            conf.random_seed = collection["random_seed"].as<unsigned long>();
            conf.max_batch_count = collection["max_batch_count"].as<unsigned int>();
            conf.batch_delay_ms = collection["batch_delay_ms"].as<unsigned int>();
        });
        return conf;
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