#!/usr/bin/env python3
"""
Test script for the recommendation microservice.

Tests three scenarios:
1. User without personal recommendations (new user)
2. User with personal recommendations but without online history
3. User with personal recommendations and online history
"""

import requests
import pandas as pd
import sys

# Service configuration
SERVICE_URL = "http://localhost:8000"
TIMEOUT = 30


def print_separator(title: str):
    """Print a formatted separator"""
    print("\n" + "=" * 80)
    print(f" {title} ")
    print("=" * 80)


def print_subsection(title: str):
    """Print a formatted subsection"""
    print(f"\n--- {title} ---")


def test_service_health():
    """Test if the service is running"""
    print_separator("SERVICE HEALTH CHECK")

    try:
        response = requests.get(f"{SERVICE_URL}/", timeout=TIMEOUT)

        if response.status_code == 200:
            print("✅ Service is running")
            print(f"Response: {response.json()}")
            return True
        else:
            print(
                f"❌ Service health check failed with status {response.status_code}")
            return False

    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to service: {e}")
        print("Please make sure the recommendation service is running on http://localhost:8000")
        return False


def get_service_stats():
    """Get and display service statistics"""
    print_subsection("Service Statistics")

    try:
        response = requests.get(f"{SERVICE_URL}/stats", timeout=TIMEOUT)

        if response.status_code == 200:
            stats = response.json()
            print(f"Total tracks in catalog: {stats['total_tracks']:,}")
            print(f"Total interactions: {stats['total_interactions']:,}")
            print(f"Active online users: {stats['active_online_users']}")
            print(
                f"Total online interactions: {stats['total_online_interactions']}")
        else:
            print(f"Failed to get stats: {response.status_code}")

    except requests.exceptions.RequestException as e:
        print(f"Error getting stats: {e}")


def find_test_users():
    """Find suitable test users from the data"""
    print_subsection("Finding Test Users")

    try:
        # Load interactions to find users with different interaction patterns
        interactions_df = pd.read_parquet("interactions.parquet")

        # Find user with many interactions (existing user)
        user_interaction_counts = interactions_df['user_id'].value_counts()
        # User with most interactions
        existing_user_id = int(user_interaction_counts.index[0])

        # Find user with few interactions
        medium_user_id = int(user_interaction_counts.index[len(
            user_interaction_counts)//2])  # Middle user

        # Use a non-existent user ID for new user test
        max_user_id = int(interactions_df['user_id'].max())
        new_user_id = max_user_id + 1000

        print(f"New user ID (no history): {new_user_id}")
        print(
            f"Medium user ID ({user_interaction_counts[medium_user_id]} interactions): {medium_user_id}")
        print(
            f"Existing user ID ({user_interaction_counts[existing_user_id]} interactions): {existing_user_id}")

        return new_user_id, medium_user_id, existing_user_id

    except Exception as e:
        print(f"Error finding test users: {e}")
        # Use default user IDs if file reading fails
        return 999999, 12345, 1


def test_user_without_recommendations(user_id: int):
    """Test scenario 1: User without personal recommendations (new user)"""
    print_separator("TEST 1: USER WITHOUT PERSONAL RECOMMENDATIONS")
    print(f"Testing with new user ID: {user_id}")

    # Check user history (should be empty)
    print_subsection("Checking User History")
    try:
        response = requests.get(
            f"{SERVICE_URL}/user_history/{user_id}", timeout=TIMEOUT)

        if response.status_code == 200:
            history = response.json()
            print(
                f"User {user_id} interaction count: {history['interaction_count']}")
            print(f"Recent tracks count: {len(history['recent_tracks'])}")
        else:
            print(f"Failed to get user history: {response.status_code}")

    except requests.exceptions.RequestException as e:
        print(f"Ошибка получения истории пользователя: {e}")

    # Get recommendations
    print_subsection("Getting Recommendations")
    try:
        request_data = {
            "user_id": user_id,
            "num_recommendations": 10
        }

        response = requests.post(
            f"{SERVICE_URL}/recommend",
            json=request_data,
            timeout=TIMEOUT
        )

        if response.status_code == 200:
            recommendations = response.json()
            print(
                f"✅ Successfully got {len(recommendations['recommendations'])} recommendations")
            print(f"Strategy used: {recommendations['strategy_used']}")
            print(f"Timestamp: {recommendations['timestamp']}")

            print("\nTop 5 recommendations:")
            for i, rec in enumerate(recommendations['recommendations'][:5]):
                print(f"  {i+1}. Track ID: {rec['track_id']}, Score: {rec['score']:.2f}, "
                      f"Strategy: {rec['strategy']}, Name: {rec['track_name'][:30]}")

            return recommendations

        else:
            print(f"❌ Failed to get recommendations: {response.status_code}")
            print(f"Response: {response.text}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"❌ Error getting recommendations: {e}")
        return None


def test_user_with_offline_history_only(user_id: int):
    """Test scenario 2: User with personal recommendations but without online history"""
    print_separator("TEST 2: USER WITH OFFLINE HISTORY ONLY")
    print(f"Testing with user ID: {user_id}")

    # Check user history
    print_subsection("Checking User History")
    try:
        response = requests.get(
            f"{SERVICE_URL}/user_history/{user_id}", timeout=TIMEOUT)

        if response.status_code == 200:
            history = response.json()
            print(
                f"User {user_id} interaction count: {history['interaction_count']}")
            print(f"Recent tracks count: {len(history['recent_tracks'])}")

            if len(history['recent_tracks']) > 0:
                print(f"Sample recent tracks: {history['recent_tracks'][:5]}")
        else:
            print(f"Failed to get user history: {response.status_code}")

    except requests.exceptions.RequestException as e:
        print(f"Ошибка получения истории пользователя: {e}")

    # Get recommendations
    print_subsection("Getting Recommendations")
    try:
        request_data = {
            "user_id": user_id,
            "num_recommendations": 10
        }

        response = requests.post(
            f"{SERVICE_URL}/recommend",
            json=request_data,
            timeout=TIMEOUT
        )

        if response.status_code == 200:
            recommendations = response.json()
            print(
                f"✅ Successfully got {len(recommendations['recommendations'])} recommendations")
            print(f"Strategy used: {recommendations['strategy_used']}")

            print("\nTop 5 recommendations:")
            for i, rec in enumerate(recommendations['recommendations'][:5]):
                print(f"  {i+1}. Track ID: {rec['track_id']}, Score: {rec['score']:.2f}, "
                      f"Strategy: {rec['strategy']}, Name: {rec['track_name'][:30]}")

            return recommendations

        else:
            print(f"❌ Failed to get recommendations: {response.status_code}")
            print(f"Response: {response.text}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"❌ Error getting recommendations: {e}")
        return None


def test_user_with_online_history(user_id: int):
    """Test scenario 3: User with personal recommendations and online history"""
    print_separator("TEST 3: USER WITH OFFLINE + ONLINE HISTORY")
    print(f"Testing with user ID: {user_id}")

    # First check initial user history
    print_subsection("Initial User History")
    try:
        response = requests.get(
            f"{SERVICE_URL}/user_history/{user_id}", timeout=TIMEOUT)

        if response.status_code == 200:
            history = response.json()
            print(
                f"User {user_id} initial interaction count: {history['interaction_count']}")
            print(
                f"Initial recent tracks count: {len(history['recent_tracks'])}")
    except requests.exceptions.RequestException as e:
        print(f"Error getting initial user history: {e}")

    # Simulate online interactions
    print_subsection("Simulating Online Interactions")

    # Simulate some track interactions
    test_track_ids = [100001, 100002, 100003, 100004, 100005]

    for track_id in test_track_ids:
        try:
            response = requests.post(
                f"{SERVICE_URL}/track_interaction",
                params={"user_id": user_id, "track_id": track_id},
                timeout=TIMEOUT
            )

            if response.status_code == 200:
                print(
                    f"✅ Tracked interaction: User {user_id} -> Track {track_id}")
            else:
                print(f"❌ Failed to track interaction: {response.status_code}")

        except requests.exceptions.RequestException as e:
            print(f"❌ Error tracking interaction: {e}")

    # Check updated user history
    print_subsection("Updated User History")
    try:
        response = requests.get(
            f"{SERVICE_URL}/user_history/{user_id}", timeout=TIMEOUT)

        if response.status_code == 200:
            history = response.json()
            print(
                f"User {user_id} updated interaction count: {history['interaction_count']}")
            print(
                f"Updated recent tracks count: {len(history['recent_tracks'])}")
            print(f"Recent online tracks: {history['recent_tracks'][:10]}")
    except requests.exceptions.RequestException as e:
        print(f"Error getting updated user history: {e}")

    # Get recommendations with online history
    print_subsection("Getting Recommendations with Online History")
    try:
        request_data = {
            "user_id": user_id,
            "num_recommendations": 10
        }

        response = requests.post(
            f"{SERVICE_URL}/recommend",
            json=request_data,
            timeout=TIMEOUT
        )

        if response.status_code == 200:
            recommendations = response.json()
            print(
                f"✅ Successfully got {len(recommendations['recommendations'])} recommendations")
            print(f"Strategy used: {recommendations['strategy_used']}")

            print("\nTop 5 recommendations:")
            for i, rec in enumerate(recommendations['recommendations'][:5]):
                print(f"  {i+1}. Track ID: {rec['track_id']}, Score: {rec['score']:.2f}, "
                      f"Strategy: {rec['strategy']}, Name: {rec['track_name'][:30]}")

            return recommendations

        else:
            print(f"❌ Failed to get recommendations: {response.status_code}")
            print(f"Response: {response.text}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"❌ Error getting recommendations: {e}")
        return None


def analyze_results(test1_result, test2_result, test3_result):
    """Analyze and compare test results"""
    print_separator("TEST RESULTS ANALYSIS")

    results = [
        ("New User (No History)", test1_result),
        ("User with Offline History", test2_result),
        ("User with Online + Offline History", test3_result)
    ]

    for test_name, result in results:
        print_subsection(test_name)

        if result:
            strategy = result.get('strategy_used', 'Unknown')
            num_recs = len(result.get('recommendations', []))

            print(f"Strategy Used: {strategy}")
            print(f"Number of Recommendations: {num_recs}")

            # Analyze recommendation strategies
            if result.get('recommendations'):
                strategies = {}
                for rec in result['recommendations'][:10]:  # First 10 recommendations
                    rec_strategy = rec.get('strategy', 'unknown')
                    strategies[rec_strategy] = strategies.get(
                        rec_strategy, 0) + 1

                print("Recommendation Strategies Distribution:")
                for strat, count in strategies.items():
                    print(f"  {strat}: {count}")

        else:
            print("❌ Test failed - no results to analyze")

    print_subsection("Summary")
    print("Expected behavior:")
    print("1. New user should get popular recommendations only")
    print("2. User with offline history should get collaborative + content-based mix")
    print("3. User with online history should get collaborative + content-based + popular mix")


def main():
    """Main test function"""
    print_separator("RECOMMENDATION SERVICE TEST SUITE")
    print("This script tests the recommendation microservice with different user scenarios.")
    print(f"Service URL: {SERVICE_URL}")

    # Test service health
    if not test_service_health():
        print("\n❌ Service is not available. Please start the service first:")
        print("python recommendations_service.py")
        sys.exit(1)

    # Get service statistics
    get_service_stats()

    # Find test users
    new_user_id, medium_user_id, existing_user_id = find_test_users()

    print(
        f"\nRunning tests with users: {new_user_id}, {medium_user_id}, {existing_user_id}")

    # Run tests
    test1_result = test_user_without_recommendations(new_user_id)
    test2_result = test_user_with_offline_history_only(medium_user_id)
    test3_result = test_user_with_online_history(existing_user_id)

    # Analyze results
    analyze_results(test1_result, test2_result, test3_result)

    print_separator("TESTING COMPLETED")

    # Summary
    successful_tests = sum(
        [1 for result in [test1_result, test2_result, test3_result] if result is not None])
    print(f"Tests completed: {successful_tests}/3 successful")

    if successful_tests == 3:
        print("✅ All tests passed successfully!")
    else:
        print("❌ Some tests failed. Check the logs above for details.")


if __name__ == "__main__":
    main()
