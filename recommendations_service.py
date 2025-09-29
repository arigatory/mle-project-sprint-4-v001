from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import uvicorn
from contextlib import asynccontextmanager

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load data on startup"""
    if not load_data():
        logger.error(
            "Не удалось загрузить данные. Сервис может работать некорректно.")
    yield


app = FastAPI(
    title="Recommendation Service",
    version="1.0.0",
    lifespan=lifespan
)

# Глобальные переменные для данных и моделей
tracks_df = None
interactions_df = None
catalog_names_df = None
user_item_matrix = None
track_popularity = None
als_model = None
user_encoder = None
item_encoder = None
lightfm_model = None

# Pydantic модели для API


class RecommendationRequest(BaseModel):
    user_id: int
    num_recommendations: int = 100


class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: List[Dict[str, Any]]
    strategy_used: str
    timestamp: datetime


class UserHistory(BaseModel):
    user_id: int
    recent_tracks: List[int] = []
    interaction_count: int = 0


# Хранение онлайн-истории в памяти
user_online_history: Dict[int, List[int]] = {}


def load_data():
    """Load data files"""
    global tracks_df, interactions_df, catalog_names_df, track_popularity

    try:
        logger.info("Загрузка файлов данных...")

        # Загрузить parquet файлы
        tracks_df = pd.read_parquet("tracks.parquet")
        interactions_df = pd.read_parquet("interactions.parquet")
        catalog_names_df = pd.read_parquet("catalog_names.parquet")

        # Вычислить популярность треков для офлайн-рекомендаций
        track_popularity = interactions_df['track_id'].value_counts().to_dict()

        logger.info(
            f"Загружено {len(tracks_df)} треков, {len(interactions_df)} взаимодействий")
        return True
    except Exception as e:
        logger.error(f"Ошибка загрузки данных: {e}")
        return False


def get_user_history(user_id: int) -> UserHistory:
    """Get user interaction history"""
    try:
        # Get offline history from interactions dataset
        user_interactions = interactions_df[interactions_df['user_id'] == user_id]
        offline_tracks = user_interactions['track_id'].tolist()

        # Get online history (recent session)
        online_tracks = user_online_history.get(user_id, [])

        # Combine histories
        # Last 50 offline + all online
        recent_tracks = online_tracks + offline_tracks[-50:]
        # Remove duplicates, preserve order
        recent_tracks = list(dict.fromkeys(recent_tracks))

        return UserHistory(
            user_id=user_id,
            recent_tracks=recent_tracks,
            interaction_count=len(user_interactions) + len(online_tracks)
        )
    except Exception as e:
        logger.error(f"Ошибка получения истории пользователя: {e}")
        return UserHistory(user_id=user_id)


def get_popular_recommendations(n_recommendations: int = 100) -> List[Dict[str, Any]]:
    """Get popular track recommendations (fallback strategy)"""
    try:
        top_tracks = sorted(track_popularity.items(),
                            key=lambda x: x[1], reverse=True)
        recommendations = []

        for i, (track_id, popularity) in enumerate(top_tracks[:n_recommendations]):
            track_info = tracks_df[tracks_df['track_id'] == track_id].iloc[0] if len(
                tracks_df[tracks_df['track_id'] == track_id]) > 0 else None

            rec = {
                "track_id": int(track_id),
                "score": float(popularity),
                "rank": i + 1,
                "strategy": "popular",
                "track_name": str(track_info['name']) if track_info is not None and 'name' in track_info else "Unknown",
                "artists": str(track_info['artists']) if track_info is not None and 'artists' in track_info else "Unknown"
            }
            recommendations.append(rec)

        return recommendations
    except Exception as e:
        logger.error(f"Ошибка получения популярных рекомендаций: {e}")
        return []


def get_collaborative_recommendations(
        user_id: int,
        user_history: UserHistory,
        n_recommendations: int = 100
) -> List[Dict[str, Any]]:
    """Get collaborative filtering recommendations (mock implementation)"""
    try:
        # This is a simplified collaborative filtering based on user similarity
        user_tracks = set(user_history.recent_tracks)

        # Find users with similar listening history
        similar_users = []
        # Уменьшаем выборку для производительности
        for other_user in interactions_df['user_id'].unique()[:100]:
            if other_user == user_id:
                continue

            other_tracks = set(
                interactions_df[interactions_df['user_id'] == other_user]['track_id'].tolist())

            # Быстрая проверка - если нет пересечений, пропускаем
            if not user_tracks.intersection(other_tracks):
                continue

            # Calculate Jaccard similarity
            intersection = len(user_tracks.intersection(other_tracks))
            union = len(user_tracks.union(other_tracks))

            if union > 0:
                similarity = intersection / union
                if similarity > 0.01:  # Понижен порог для большего количества похожих пользователей
                    similar_users.append((other_user, similarity))

        # Get recommendations from similar users
        similar_users.sort(key=lambda x: x[1], reverse=True)
        recommended_tracks = {}

        # Top 10 similar users
        for similar_user, similarity in similar_users[:10]:
            similar_user_tracks = interactions_df[interactions_df['user_id']
                                                  == similar_user]['track_id'].value_counts()

            # Top tracks from similar user
            for track_id, count in similar_user_tracks.head(20).items():
                if track_id not in user_tracks:  # Не рекомендовать уже прослушанные треки
                    score = similarity * count
                    if track_id in recommended_tracks:
                        recommended_tracks[track_id] += score
                    else:
                        recommended_tracks[track_id] = score

        # Если нет рекомендаций, вернуть пустой список
        if not recommended_tracks:
            return []

        # Sort recommendations by score
        sorted_recommendations = sorted(
            recommended_tracks.items(), key=lambda x: x[1], reverse=True)

        recommendations = []
        for i, (track_id, score) in enumerate(sorted_recommendations[:n_recommendations]):
            track_info = tracks_df[tracks_df['track_id'] == track_id].iloc[0] if len(
                tracks_df[tracks_df['track_id'] == track_id]) > 0 else None

            rec = {
                "track_id": int(track_id),
                "score": float(score),
                "rank": i + 1,
                "strategy": "collaborative",
                "track_name": str(track_info['name']) if track_info is not None and 'name' in track_info else "Unknown",
                "artists": str(track_info['artists']) if track_info is not None and 'artists' in track_info else "Unknown"
            }
            recommendations.append(rec)

        return recommendations

    except Exception as e:
        logger.error(f"Ошибка получения коллаборативных рекомендаций: {e}")
        return []


def get_content_based_recommendations(
        user_id: int,
        user_history: UserHistory,
        n_recommendations: int = 100
) -> List[Dict[str, Any]]:
    """Get content-based recommendations using track features"""
    try:
        if not user_history.recent_tracks:
            return []

        # Get genres/features of user's recently listened tracks
        user_tracks_info = tracks_df[tracks_df['track_id'].isin(
            user_history.recent_tracks)]

        if user_tracks_info.empty:
            return []

        # Extract user preferences (simplified - using artists and albums as content features)
        user_artists = set()
        user_albums = set()

        for _, track in user_tracks_info.iterrows():
            if 'artists' in track and pd.notna(track['artists']):
                user_artists.update(str(track['artists']).split(', '))
            if 'albums' in track and pd.notna(track['albums']):
                user_albums.update(str(track['albums']).split(', '))

        # Find tracks with similar content
        content_scores = {}
        listened_tracks = set(user_history.recent_tracks)

        # Ограничиваем количество треков для анализа (для производительности)
        sample_tracks = tracks_df.sample(
            min(10000, len(tracks_df)), random_state=42)

        for _, track in sample_tracks.iterrows():
            try:
                track_id = track['track_id']
                if track_id in listened_tracks:
                    continue

                score = 0

                # Artist similarity
                if 'artists' in track and pd.notna(track['artists']):
                    try:
                        track_artists = set(str(track['artists']).split(', '))
                        artist_overlap = len(
                            user_artists.intersection(track_artists))
                        score += artist_overlap * 2
                    except:
                        pass

                # Album similarity
                if 'albums' in track and pd.notna(track['albums']):
                    try:
                        track_albums = set(str(track['albums']).split(', '))
                        album_overlap = len(
                            user_albums.intersection(track_albums))
                        score += album_overlap * 1
                    except:
                        pass

                # Add popularity boost
                popularity = track_popularity.get(track_id, 0)
                score += np.log(popularity + 1) * 0.1

                if score > 0:
                    content_scores[track_id] = score
            except Exception as e:
                logger.error(f"Ошибка обработки трека {track_id}: {e}")
                continue

        # Sort by score
        sorted_recommendations = sorted(
            content_scores.items(), key=lambda x: x[1], reverse=True)

        recommendations = []
        for i, (track_id, score) in enumerate(sorted_recommendations[:n_recommendations]):
            track_info = tracks_df[tracks_df['track_id'] == track_id].iloc[0]

            rec = {
                "track_id": int(track_id),
                "score": float(score),
                "rank": i + 1,
                "strategy": "content_based",
                "track_name": str(track_info['name']) if 'name' in track_info else "Unknown",
                "artists": str(track_info['artists']) if 'artists' in track_info else "Unknown"
            }
            recommendations.append(rec)

        return recommendations

    except Exception as e:
        logger.error(f"Ошибка получения контентных рекомендаций: {e}")
        return []


def mix_recommendations(user_id: int, user_history: UserHistory, n_recommendations: int = 100) -> List[Dict[str, Any]]:
    """
    Mix online and offline recommendations using a hybrid strategy

    Strategy:
    1. For new users (no history): 100% popular tracks
    2. For users with offline history only: 70% collaborative + 30% content-based
    3. For users with online history: 50% collaborative + 30% content-based + 20% popular
    """

    try:
        online_tracks = user_online_history.get(user_id, [])
        has_offline_history = user_history.interaction_count > len(
            online_tracks)
        has_online_history = len(online_tracks) > 0

        # Determine mixing strategy
        if not has_offline_history and not has_online_history:
            # New user - only popular recommendations
            strategy_used = "popular_only"
            recommendations = get_popular_recommendations(n_recommendations)

        elif has_offline_history and not has_online_history:
            # User with offline history only
            strategy_used = "offline_hybrid"

            # 70% collaborative, 30% content-based
            collab_recs = get_collaborative_recommendations(
                user_id, user_history, int(n_recommendations * 0.7))
            content_recs = get_content_based_recommendations(
                user_id, user_history, int(n_recommendations * 0.3))

            # Объединить рекомендации
            recommendations = collab_recs + content_recs

        else:
            # User with online history
            strategy_used = "online_offline_hybrid"

            # 50% collaborative, 30% content-based, 20% popular
            collab_recs = get_collaborative_recommendations(
                user_id, user_history, int(n_recommendations * 0.5))
            content_recs = get_content_based_recommendations(
                user_id, user_history, int(n_recommendations * 0.3))
            popular_recs = get_popular_recommendations(
                int(n_recommendations * 0.2))

            # Объединить рекомендации
            recommendations = collab_recs + content_recs + popular_recs

        # Remove duplicates and re-rank
        seen_tracks = set()
        final_recommendations = []

        for rec in recommendations:
            if rec['track_id'] not in seen_tracks and rec['track_id'] not in user_history.recent_tracks:
                seen_tracks.add(rec['track_id'])
                rec['rank'] = len(final_recommendations) + 1
                final_recommendations.append(rec)

                if len(final_recommendations) >= n_recommendations:
                    break

        # If we don't have enough recommendations, fill with popular tracks
        if len(final_recommendations) < n_recommendations:
            additional_popular = get_popular_recommendations(
                n_recommendations - len(final_recommendations))
            for rec in additional_popular:
                if rec['track_id'] not in seen_tracks and rec['track_id'] not in user_history.recent_tracks:
                    seen_tracks.add(rec['track_id'])
                    rec['rank'] = len(final_recommendations) + 1
                    final_recommendations.append(rec)

                    if len(final_recommendations) >= n_recommendations:
                        break

        # Add strategy information
        for rec in final_recommendations:
            rec['mixed_strategy'] = strategy_used

        return final_recommendations

    except Exception as e:
        logger.error(f"Ошибка смешивания рекомендаций: {e}")
        # Fallback to popular recommendations
        fallback_recs = get_popular_recommendations(n_recommendations)
        for rec in fallback_recs:
            rec['mixed_strategy'] = "fallback_popular"
        return fallback_recs


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Сервис рекомендаций работает", "status": "healthy"}


@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """Get recommendations for a user"""
    try:
        # Get user history
        user_history = get_user_history(request.user_id)

        # Get mixed recommendations
        recommendations = mix_recommendations(
            request.user_id,
            user_history,
            request.num_recommendations
        )

        # Извлекаем стратегию из рекомендаций
        strategy_used = recommendations[0].get(
            'mixed_strategy', 'unknown') if recommendations else 'empty'

        return RecommendationResponse(
            user_id=request.user_id,
            recommendations=recommendations,
            strategy_used=strategy_used,
            timestamp=datetime.now()
        )

    except Exception as e:
        logger.error(
            f"Ошибка генерации рекомендаций for user {request.user_id}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Ошибка генерации рекомендаций: {str(e)}")


@app.post("/track_interaction")
async def track_user_interaction(user_id: int, track_id: int):
    """Track user interaction (online history)"""
    try:
        if user_id not in user_online_history:
            user_online_history[user_id] = []

        # Add track to user's online history (keep last 50 interactions)
        if track_id not in user_online_history[user_id]:
            user_online_history[user_id].append(track_id)

            # Keep only last 50 interactions
            if len(user_online_history[user_id]) > 50:
                user_online_history[user_id] = user_online_history[user_id][-50:]

        return {"message": "Взаимодействие успешно отслежено", "user_id": user_id, "track_id": track_id}

    except Exception as e:
        logger.error(f"Ошибка отслеживания взаимодействия: {e}")
        raise HTTPException(
            status_code=500, detail=f"Ошибка отслеживания взаимодействия: {str(e)}")


@app.get("/user_history/{user_id}")
async def get_user_history_endpoint(user_id: int):
    """Get user interaction history"""
    try:
        history = get_user_history(user_id)
        return history
    except Exception as e:
        logger.error(f"Ошибка получения истории пользователя: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error getting user history: {str(e)}")


@app.get("/stats")
async def get_service_stats():
    """Get service statistics"""
    try:
        stats = {
            "total_tracks": len(tracks_df) if tracks_df is not None else 0,
            "total_interactions": len(interactions_df) if interactions_df is not None else 0,
            "active_online_users": len(user_online_history),
            "total_online_interactions": sum(len(history) for history in user_online_history.values())
        }
        return stats
    except Exception as e:
        logger.error(f"Ошибка получения статистики: {e}")
        raise HTTPException(
            status_code=500, detail=f"Ошибка получения статистики: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
