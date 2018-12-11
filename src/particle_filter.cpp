/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

const double epsilon = 1e-5;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	if (!is_initialized) {

		default_random_engine gen;
		num_particles = 100;

		normal_distribution<double> dist_x(x, std[0]);
		normal_distribution<double> dist_y(y, std[1]);
		normal_distribution<double> dist_theta(theta, std[2]);

		for (int i = 0; i < num_particles; ++i) {
			Particle particle;
			particle.id = i;
			particle.x = dist_x(gen);
      particle.y = dist_y(gen);
      particle.theta = dist_theta(gen);
			particle.weight = 1;
			particles.push_back(particle);

			weights.push_back(1);
		}

		is_initialized = true;
	}
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	default_random_engine gen;

	bool use_linear_model = (fabs(yaw_rate) < epsilon); // use linear model if yaw_rate is close to zero

	auto velocity_to_yaw = use_linear_model ? 1 : (velocity/yaw_rate);

	for (auto& particle : particles) {

		if (use_linear_model) {
			particle.x += velocity*delta_t*cos(particle.theta);
			particle.y += velocity*delta_t*sin(particle.theta);
		}
		else {
			particle.x += velocity_to_yaw*(sin(particle.theta + yaw_rate*delta_t) - sin(particle.theta));
			particle.y += velocity_to_yaw*(cos(particle.theta) - cos(particle.theta + yaw_rate*delta_t));
			particle.theta += yaw_rate*delta_t;
		}

		// add noise
		normal_distribution<double> dist_x(particle.x, std_pos[0]);
		normal_distribution<double> dist_y(particle.y, std_pos[1]);
		normal_distribution<double> dist_theta(particle.theta, std_pos[2]);

		particle.x = dist_x(gen);
		particle.y = dist_y(gen);
		particle.theta = dist_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.
	for (auto& observation: observations) {
		auto min_distance = INFINITY;
		auto id = 0;

		for (const auto& landmark: predicted) {
			float distance = dist(observation.x, observation.y, landmark.x, landmark.y);

			if (distance < min_distance) {
				observation.id = landmark.id;
				min_distance = distance;
			}
		}

		//cout << "observation x = " << observation.x << "; observation y = " << observation.y << "; observation id = " << observation.id << endl;
		//cout << "landmark x = " << predicted[observation.id-1].x << "; landmark y = " << predicted[observation.id-1].y << "; landmark id = " << predicted[observation.id-1].id << endl;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	double std_x = std_landmark[0];
	double std_y = std_landmark[1];

	weights.clear();

	double weights_sum = 0;

	auto gauss_norm = 1.0/(2.0*M_PI*std_x*std_y);

	// build the list of landmarks
	std::vector<LandmarkObs> landmarks;

	for (const auto& landmark : map_landmarks.landmark_list) {
		LandmarkObs landm;
		landm.x = landmark.x_f;
		landm.y = landmark.y_f;
		landm.id = landmark.id_i;
		landmarks.push_back(landm);
	}

	for (auto& particle : particles) {
		// transform the observations to map coordinate system
		std::vector<LandmarkObs> particle_obs;

		for (const auto& observation: observations) {
			LandmarkObs obs;
			obs.x = particle.x + cos(particle.theta)*observation.x - sin(particle.theta)*observation.y;
			obs.y = particle.y + sin(particle.theta)*observation.x + cos(particle.theta)*observation.y;

			particle_obs.push_back(obs);
		}

		// do the data association
		dataAssociation(landmarks, particle_obs);

		// update weights
		particle.weight = 1.0;

		for (const auto& observation : particle_obs) {
			auto landmark = landmarks[observation.id-1];
			auto exponent = (pow(observation.x-landmark.x, 2)/(2.0*pow(std_x, 2)) + pow(observation.y-landmark.y, 2)/(2.0*pow(std_y, 2)));
			//cout << "exponent = " << exponent << endl;

			double p = gauss_norm*exp(-exponent);
			//cout << "p = " << p << endl;

			particle.weight *= p;
		}

		weights.push_back(particle.weight);

		weights_sum += particle.weight;
	}

	// normalize weights
	for (auto& weight: weights) {
		weight /= weights_sum;
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	/*
	// for debugging purposes only
	cout << "resample" << endl;
	for (const auto& weight : weights) {
		cout << weight << " ";
	}

	cout << endl;
	*/

	std::random_device rd;
  std::mt19937 gen(rd());

	std::discrete_distribution<> d(begin(weights), end(weights));

	std::vector<Particle> sampled_particles;

	for (int i = 0; i < particles.size(); ++i) {
		sampled_particles.push_back(particles[d(gen)]);
	}

	// update the particles collection based on the resampled particles
	particles = sampled_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
