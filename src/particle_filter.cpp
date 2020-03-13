/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <array>

#include "helper_functions.h"

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  
    num_particles = 10;  // TODO: Set the number of particles    
    weights.resize(num_particles);

    dist_x = std::normal_distribution<double>(0, std[0]);
    dist_y = std::normal_distribution<double>(0, std[1]);
    dist_theta = std::normal_distribution<double>(0, std[2]);

    for(int i = 0; i < num_particles ; i++)
    {
      Particle particle;

      particle.x = dist_x(gen) + x;
      particle.y = dist_y(gen) + y;
      particle.theta = dist_theta(gen) + theta;
      particle.weight = 1.0;

      particles.push_back(particle);

    }
    is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */

  for (auto &particle : particles)
  {
    if (yaw_rate == 0)
    {
      particle.x += velocity * delta_t * cos(particle.theta);
      particle.y += velocity * delta_t * sin(particle.theta);
    }
    else
    {
      particle.x += (velocity * (sin(particle.theta + yaw_rate * delta_t) - sin(particle.theta))) / yaw_rate;
      particle.y += (velocity * (cos(particle.theta) - cos(particle.theta + yaw_rate * delta_t))) / yaw_rate;
    }
    particle.theta += yaw_rate * delta_t;
    particle.x += dist_x(gen);
    particle.y += dist_y(gen);
    particle.theta += dist_theta(gen);
  }

}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */

  for (auto &observation : observations)
  {
    double minimum_distance = std::numeric_limits<double>::infinity();
    for (auto &prediction : predicted)
    {
      double distance = dist(prediction.x, prediction.y, observation.x, observation.y);

      if (distance < minimum_distance)
      {
        observation.id = prediction.id;
        minimum_distance = distance;
      }
    }
  }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  double total_weight = 0;

  for (auto &particle : particles)
  {
    auto trans_observations(observations);

    for (auto &observation : trans_observations)
    {
      auto obs_x = observation.x;
      auto obs_y = observation.y;
      observation.x = particle.x + (cos(particle.theta) * obs_x - sin(particle.theta) * obs_y);
      observation.y = particle.y + (sin(particle.theta) * obs_x + cos(particle.theta) * obs_y);
    }

    std::vector<LandmarkObs> visible_landmarks;
    for (const auto &landmark : map_landmarks.landmark_list)
    {
      if (dist(particle.x, particle.y, landmark.x_f, landmark.y_f) < sensor_range)
      {
        visible_landmarks.push_back(LandmarkObs{landmark.id_i, landmark.x_f, landmark.y_f});
      }
    }

    dataAssociation(visible_landmarks, trans_observations);
    particle.weight = 1.0;
    for (const auto &observation : trans_observations)
    {
      for (const auto &landmark : visible_landmarks)
      {
        if (landmark.id == observation.id)
        {
          particle.weight *= calWeight(std_landmark, observation.x, observation.y, landmark.x, landmark.y);          
        }
      }
    }
    total_weight += particle.weight;
  }

  int i = 0;
  for (vector<Particle>::iterator itr = particles.begin(); itr != particles.end(); itr++)
  {
    itr->weight /= total_weight;
    weights[i] = (itr->weight);
    // std::cout <<weights[i] <<"\t";
    i++;
  }
  // std::cout<<std::endl;
 
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  std::vector<Particle> resampled_particles(num_particles);
  std::discrete_distribution<> random_index(weights.begin(), weights.end());

  for (auto &new_particle : resampled_particles)
  {
    new_particle = particles[random_index(gen)];
  }
  particles = resampled_particles;

}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}