#!/usr/bin/env python3
"""WFS NFL — FanDuel Injury News Ingest V2.

Secondary injury/news ingest only. FanDuel Research never creates WFS identities and
never overrides structured injury authority. Exact normalized identity matching only.
"""
from __future__ import annotations

import hashlib
import html
import json
import re
import sqlite3
import unicodedata
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "data" / "nfl.db"
CSV_DIR = ROOT / "data" / "csv"
INGEST_CSV = CSV_DIR / "fanduel_injury_ingest_current.csv"
QUARANTINE_CSV = CSV_DIR / "fanduel_injury_quarantine.csv"
URL = "https://www.fanduel.com/research/nfl/player-news/injuries"
GRAPHQL_URL = "https://www.fanduel.com/research/api/graphql"
SOURCE = "FANDUEL_RESEARCH"
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140 Safari/537.36"
PAGE_LIMIT = 10
MAX_PAGES = 500
ET = ZoneInfo("America/New_York")
WINDOW_BEFORE = timedelta(days=7)
WINDOW_AFTER = timedelta(hours=24)

PLAYER_NEWS_QUERY = r"""
query getPlayerNewsQuery($filter: ShortFormSearchInput!) {
  getShortForms(filter: $filter) {
    pageInfo { cursor hasNextPage }
    shortForms {
      cursor
      entity {
        id slug title fact analysis quant
        newsType { name enum }
        primaryRef {
          ... on Player {
            identifier name number position playerPageUrl
            team { name }
          }
        }
        firstPublishedAt lastPublishedAt description
      }
    }
  }
}
""".strip()

TEAM_ALIASES = {
    "ARIZONA CARDINALS":"ARI","ATLANTA FALCONS":"ATL","BALTIMORE RAVENS":"BAL","BUFFALO BILLS":"BUF",
    "CAROLINA PANTHERS":"CAR","CHICAGO BEARS":"CHI","CINCINNATI BENGALS":"CIN","CLEVELAND BROWNS":"CLE",
    "DALLAS COWBOYS":"DAL","DENVER BRONCOS":"DEN","DETROIT LIONS":"DET","GREEN BAY PACKERS":"GB",
    "HOUSTON TEXANS":"HOU","INDIANAPOLIS COLTS":"IND","JACKSONVILLE JAGUARS":"JAX","KANSAS CITY CHIEFS":"KC",
    "LAS VEGAS RAIDERS":"LV","LOS ANGELES CHARGERS":"LAC","LOS ANGELES RAMS":"LA","MIAMI DOLPHINS":"MIA",
    "MINNESOTA VIKINGS":"MIN","NEW ENGLAND PATRIOTS":"NE","NEW ORLEANS SAINTS":"NO","NEW YORK GIANTS":"NYG",
    "NEW YORK JETS":"NYJ","PHILADELPHIA EAGLES":"PHI","PITTSBURGH STEELERS":"PIT","SAN FRANCISCO 49ERS":"SF",
    "SEATTLE SEAHAWKS":"SEA","TAMPA BAY BUCCANEERS":"TB","TENNESSEE TITANS":"TEN","WASHINGTON COMMANDERS":"WAS",
}
DIRECT_TEAMS = {x:x for x in ["ARI","ATL","BAL","BUF","CAR","CHI","CIN","CLE","DAL","DEN","DET","GB","HOU","IND","JAX","KC","LV","LAC","LA","MIA","MIN","NE","NO","NYG","NYJ","PHI","PIT","SF","SEA","TB","TEN","WAS"]}
DIRECT_TEAMS.update({"GBP":"GB","JAC":"JAX","KAN":"KC","LVR":"LV","LAR":"LA","NEP":"NE","NOS":"NO","SFO":"SF","TAM":"TB","WSH":"WAS"})


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan","none","null","<na>"} else text


def normalize_name(value: Any) -> str:
    text = unicodedata.normalize("NFKD", clean_text(value))
    text = "".join(c for c in text if not unicodedata.combining(c)).upper()
    return re.sub(r"\s+", " ", re.sub(r"[^A-Z0-9]+", " ", text)).strip()


def normalize_team(value: Any) -> str:
    text = clean_text(value).upper()
    if not text:
        return ""
    if text in TEAM_ALIASES:
        return TEAM_ALIASES[text]
    compact = re.sub(r"[^A-Z0-9]", "", text)
    return DIRECT_TEAMS.get(compact, text)


def normalize_position(value: Any) -> str:
    text = clean_text(value).upper()
    return {"HB":"RB","FB":"RB"}.get(text, text)


def flatten_blocks(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return ""
        try:
            return flatten_blocks(json.loads(raw))
        except Exception:
            return raw
    if isinstance(value, list):
        return " ".join(filter(None, (flatten_blocks(x) for x in value))).strip()
    if isinstance(value, dict):
        if "text" in value:
            return clean_text(value.get("text"))
        return " ".join(filter(None, (flatten_blocks(x) for x in value.values()))).strip()
    return clean_text(value)


def http_get(url: str) -> requests.Response:
    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"FanDuel GET HTTP status {r.status_code}: {url}")
    return r


def fetch_next_data() -> dict:
    r = http_get(URL)
    matches = re.findall(r'<script[^>]*id=["\']__NEXT_DATA__["\'][^>]*>(.*?)</script>', r.text, flags=re.I|re.S)
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one __NEXT_DATA__ block; found {len(matches)}")
    data = json.loads(html.unescape(matches[0].strip()))
    if not isinstance(data, dict):
        raise RuntimeError("FanDuel __NEXT_DATA__ root is not an object")
    return data


def injury_entity(entity: Any) -> bool:
    return isinstance(entity, dict) and clean_text((entity.get("newsType") or {}).get("enum")).upper() == "INJURY"


def find_embedded_page(data: Any) -> tuple[list[dict], dict]:
    candidates: list[tuple[list[dict], dict]] = []
    def walk(obj: Any) -> None:
        if isinstance(obj, dict):
            gs = obj.get("getShortForms")
            if isinstance(gs, dict) and isinstance(gs.get("shortForms"), list) and isinstance(gs.get("pageInfo"), dict):
                entities = [x.get("entity") for x in gs["shortForms"] if isinstance(x, dict) and injury_entity(x.get("entity"))]
                if entities:
                    candidates.append((entities, gs["pageInfo"]))
            for v in obj.values(): walk(v)
        elif isinstance(obj, list):
            for v in obj: walk(v)
    walk(data)
    if not candidates:
        raise RuntimeError("Embedded FanDuel injury page not found")
    # The injury query is the candidate with the largest injury population.
    candidates.sort(key=lambda x: len(x[0]), reverse=True)
    return candidates[0]


def graphql_page(cursor: str) -> tuple[list[dict], dict]:
    variables = {"filter": {"afterCursor": cursor, "limit": PAGE_LIMIT, "player":{"positionAbbrev":None}, "publishedWithin":None, "shortFormNewsType":"INJURY", "sport":"NFL", "team":{"numberFireId":None}}}
    r = requests.post(GRAPHQL_URL, headers={"User-Agent":USER_AGENT,"Content-Type":"application/json"}, json={"query":PLAYER_NEWS_QUERY,"variables":variables}, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"FanDuel GraphQL HTTP status {r.status_code}")
    payload = r.json()
    if payload.get("errors"):
        raise RuntimeError(f"FanDuel GraphQL errors: {payload['errors']}")
    gs = ((payload.get("data") or {}).get("getShortForms") or {})
    rows = gs.get("shortForms")
    page_info = gs.get("pageInfo")
    if not isinstance(rows, list) or not isinstance(page_info, dict):
        raise RuntimeError("FanDuel GraphQL response contract changed")
    entities = [x.get("entity") for x in rows if isinstance(x, dict) and injury_entity(x.get("entity"))]
    return entities, page_info


def fetch_all_entities() -> tuple[list[dict], dict]:
    data = fetch_next_data()
    entities, info = find_embedded_page(data)
    raw_rows = len(entities)
    pages = 1
    boundary_dupes = 0
    unique: dict[str, dict] = {}
    for e in entities:
        event_id = clean_text(e.get("id"))
        if not event_id: raise RuntimeError("Blank FanDuel source event ID")
        unique[event_id] = e
    while bool(info.get("hasNextPage")):
        if pages >= MAX_PAGES:
            raise RuntimeError(f"Pagination exceeded MAX_PAGES={MAX_PAGES}")
        cursor = clean_text(info.get("cursor"))
        if not cursor: raise RuntimeError("hasNextPage=true with blank cursor")
        page, info = graphql_page(cursor)
        pages += 1
        raw_rows += len(page)
        for e in page:
            event_id = clean_text(e.get("id"))
            if not event_id: raise RuntimeError("Blank FanDuel source event ID")
            if event_id in unique:
                # Proven source behavior: identical event may repeat at cursor boundary.
                a = json.dumps(unique[event_id], sort_keys=True, separators=(",",":"), ensure_ascii=False)
                b = json.dumps(e, sort_keys=True, separators=(",",":"), ensure_ascii=False)
                if a != b: raise RuntimeError(f"Duplicate event ID changed payload: {event_id}")
                boundary_dupes += 1
            else:
                unique[event_id] = e
    if bool(info.get("hasNextPage")):
        raise RuntimeError("Pagination did not terminate")
    return list(unique.values()), {"pages":pages,"raw_rows":raw_rows,"unique_events":len(unique),"boundary_duplicates":boundary_dupes,"terminal":True}


def extract_player_name(entity: dict) -> str:
    primary = entity.get("primaryRef") or {}
    for key in ("name","fullName","displayName","playerName"):
        text = clean_text(primary.get(key))
        if text: return text
    title = clean_text(entity.get("title"))
    for pattern in (r"\s+Full Participant\b.*$",r"\s+Limited\b.*$",r"\s+Did Not Practice\b.*$",r"\s+Ruled Out\b.*$",r"\s+Questionable\b.*$",r"\s+Doubtful\b.*$"):
        candidate = re.sub(pattern,"",title,flags=re.I).strip()
        if candidate != title: return candidate
    return ""


def extract_team(entity: dict) -> str:
    team = (entity.get("primaryRef") or {}).get("team") or {}
    return clean_text(team.get("name")) if isinstance(team, dict) else clean_text(team)


def extract_position(entity: dict) -> str:
    return clean_text((entity.get("primaryRef") or {}).get("position"))


def extract_fd_player_id(entity: dict) -> str:
    url = clean_text((entity.get("primaryRef") or {}).get("playerPageUrl"))
    m = re.search(r"-(\d+)(?:[/?#]|$)", url)
    return m.group(1) if m else ""


def classify_practice_signal(title: str, description: str) -> str:
    text = f"{title} {description}".lower()
    if "did not participate" in text or "did not practice" in text: return "DNP"
    if "limited participant" in text or "limited in practice" in text: return "LIMITED"
    if "full participant" in text or "full participation" in text: return "FULL"
    return ""


def classify_availability_signal(title: str, description: str) -> str:
    text = f"{title} {description}".lower()
    if any(x in text for x in ("ruled out","will not play","won't play","inactive")): return "OUT_SIGNAL"
    if "doubtful" in text: return "DOUBTFUL_SIGNAL"
    if "questionable" in text: return "QUESTIONABLE_SIGNAL"
    if "expected to play" in text or "expected to suit up" in text: return "EXPECTED_TO_PLAY"
    return ""


def signal_strength(practice: str, availability: str) -> str:
    if availability in {"OUT_SIGNAL","DOUBTFUL_SIGNAL"}: return "HIGH"
    if availability in {"QUESTIONABLE_SIGNAL","EXPECTED_TO_PLAY"} or practice == "DNP": return "MEDIUM"
    return "LOW"


def parse_source_timestamp(value: Any) -> datetime | None:
    text = clean_text(value)
    if not text: return None
    try:
        dt = datetime.fromisoformat(text.replace("Z","+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None: return None
    return dt.astimezone(timezone.utc)


def explicit_weeks(text: str) -> set[int]:
    return {int(x) for x in re.findall(r"\bWeek\s+(\d{1,2})\b", text, flags=re.I)}


def extract_records(entities: list[dict]) -> pd.DataFrame:
    records=[]
    for e in entities:
        primary=e.get("primaryRef") or {}
        title=clean_text(e.get("title")); description=clean_text(e.get("description")); fact=flatten_blocks(e.get("fact"))
        if not description: description=fact
        player=extract_player_name(e); team_name=extract_team(e); pos=extract_position(e)
        practice=classify_practice_signal(title,description); availability=classify_availability_signal(title,description)
        event_id=clean_text(e.get("id"))
        records.append({
            "source_event_id":event_id,
            "signal_key":hashlib.sha256(f"{SOURCE}|{event_id}".encode()).hexdigest(),
            "fanduel_entity_id":event_id,"fanduel_player_id":extract_fd_player_id(e),"slug":clean_text(e.get("slug")),
            "player_name":player,"normalized_name":normalize_name(player),"team_name":team_name,"team":normalize_team(team_name),
            "position":normalize_position(pos),"title":title,"description":description,"fact":fact,
            "practice_signal":practice,"availability_signal":availability,"signal_strength":signal_strength(practice,availability),
            "source_timestamp":clean_text(e.get("firstPublishedAt")),"last_published_at":clean_text(e.get("lastPublishedAt")),
            "player_page_url":clean_text(primary.get("playerPageUrl")),
            "explicit_weeks":sorted(explicit_weeks(f"{title} {description} {fact}")),
        })
    return pd.DataFrame(records)


def table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {r[1] for r in conn.execute(f'PRAGMA table_info("{table}")')}


def latest_season_week(conn: sqlite3.Connection) -> tuple[int,int]:
    row=conn.execute("SELECT season,week FROM injuries WHERE season IS NOT NULL AND week IS NOT NULL ORDER BY season DESC,week DESC LIMIT 1").fetchone()
    if row is None: raise RuntimeError("Unable to determine current injury season/week")
    return int(row[0]),int(row[1])


def schedule_window(conn: sqlite3.Connection, season: int, week: int) -> tuple[datetime,datetime,set[str]]:
    rows=conn.execute("SELECT game_id,game_date,gametime,away_team,home_team FROM games WHERE season=? AND week=? ORDER BY game_date,gametime,game_id",(season,week)).fetchall()
    if not rows: raise RuntimeError("No current-week games")
    kickoffs=[]; teams=set()
    for game_id,game_date,gametime,away,home in rows:
        raw=f"{clean_text(game_date)} {clean_text(gametime)}"
        try: naive=datetime.strptime(raw,"%Y-%m-%d %H:%M")
        except ValueError as exc: raise RuntimeError(f"Invalid authoritative kickoff {game_id}: {raw}") from exc
        kickoffs.append(naive.replace(tzinfo=ET).astimezone(timezone.utc))
        teams.update({normalize_team(away),normalize_team(home)})
    return min(kickoffs)-WINDOW_BEFORE,max(kickoffs)+WINDOW_AFTER,{x for x in teams if x}


def filter_current_week(records: pd.DataFrame, week: int, start_utc: datetime, end_utc: datetime, teams: set[str]) -> tuple[pd.DataFrame,pd.DataFrame]:
    keep=[]; reasons=[]
    for row in records.itertuples(index=False):
        ts=parse_source_timestamp(row.source_timestamp)
        weeks=set(row.explicit_weeks if isinstance(row.explicit_weeks,list) else [])
        reason="CURRENT_WEEK"
        ok=True
        if ts is None: ok=False; reason="UNDATED"
        elif not (start_utc <= ts <= end_utc): ok=False; reason="OUTSIDE_SCHEDULE_WINDOW"
        elif weeks != {week}: ok=False; reason="WEEK_REFERENCE_MISMATCH_OR_MISSING"
        elif normalize_team(row.team) not in teams: ok=False; reason="TEAM_NOT_IN_CURRENT_SCHEDULE"
        keep.append(ok); reasons.append(reason)
    out=records.copy(); out["current_filter_reason"]=reasons
    return out[pd.Series(keep,index=out.index)].copy(),out[~pd.Series(keep,index=out.index)].copy()


def build_identity_index(conn: sqlite3.Connection, season: int, week: int) -> pd.DataFrame:
    frames=[]
    if {"gsis_id","full_name","team","position","season","week"}.issubset(table_columns(conn,"injuries")):
        frames.append(pd.read_sql_query("SELECT gsis_id,full_name AS player_name,team,position,'injuries' AS identity_source FROM injuries WHERE season=? AND week=?",conn,params=(season,week)))
    tables={r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "weekly_rosters" in tables:
        cols=table_columns(conn,"weekly_rosters")
        gsis=next((c for c in ("gsis_id","player_id") if c in cols),None); name=next((c for c in ("full_name","player_name","player_display_name") if c in cols),None)
        team=next((c for c in ("team","recent_team") if c in cols),None); pos=next((c for c in ("position","position_group") if c in cols),None)
        if gsis and name and team and pos and "season" in cols:
            where='"season"=?'; params=[season]
            if "week" in cols: where+=' AND "week"=?'; params.append(week)
            frames.append(pd.read_sql_query(f'SELECT "{gsis}" AS gsis_id,"{name}" AS player_name,"{team}" AS team,"{pos}" AS position,\'weekly_rosters\' AS identity_source FROM weekly_rosters WHERE {where}',conn,params=params))
    if not frames: raise RuntimeError("No authoritative identity source available")
    identity=pd.concat(frames,ignore_index=True)
    identity["gsis_id"]=identity["gsis_id"].map(clean_text); identity["player_name"]=identity["player_name"].map(clean_text)
    identity["normalized_name"]=identity["player_name"].map(normalize_name); identity["team"]=identity["team"].map(normalize_team); identity["position"]=identity["position"].map(normalize_position)
    identity=identity[identity["gsis_id"].ne("") & identity["normalized_name"].ne("")].copy()
    return identity.drop_duplicates(subset=["gsis_id","normalized_name","team","position"])


def resolve_identity(row: pd.Series, identity: pd.DataFrame) -> tuple[str,str,str]:
    name=clean_text(row.get("normalized_name")); team=normalize_team(row.get("team")); pos=normalize_position(row.get("position"))
    if not name: return "","QUARANTINE","BLANK_PLAYER_NAME"
    candidates=identity[identity["normalized_name"].eq(name)].copy()
    if candidates.empty: return "","QUARANTINE","NO_EXACT_NAME_MATCH"
    ids=sorted(set(candidates["gsis_id"].tolist()))
    if len(ids)==1:
        c=candidates[candidates["gsis_id"].eq(ids[0])]
        teams={normalize_team(x) for x in c["team"].tolist() if clean_text(x)}; positions={normalize_position(x) for x in c["position"].tolist() if clean_text(x)}
        if team and teams and team not in teams: return "","QUARANTINE","TEAM_MISMATCH"
        if pos and positions and pos not in positions: return "","QUARANTINE","POSITION_MISMATCH"
        return ids[0],"MATCHED","EXACT_UNIQUE_NAME_VALIDATED"
    filtered=candidates.copy()
    if team: filtered=filtered[filtered["team"].eq(team)]
    if pos: filtered=filtered[filtered["position"].eq(pos)]
    ids=sorted(set(filtered["gsis_id"].tolist()))
    if len(ids)==1: return ids[0],"MATCHED","EXACT_NAME_TEAM_POSITION"
    if not ids: return "","QUARANTINE","IDENTITY_CONSTRAINT_MISMATCH"
    return "","QUARANTINE","AMBIGUOUS_EXACT_IDENTITY"


def attach_identity(records: pd.DataFrame, identity: pd.DataFrame) -> pd.DataFrame:
    out=records.copy(); resolved=out.apply(lambda r:resolve_identity(r,identity),axis=1)
    out["gsis_id"]=[x[0] for x in resolved]; out["resolution_status"]=[x[1] for x in resolved]; out["resolution_detail"]=[x[2] for x in resolved]
    return out


def migrate_schema(conn: sqlite3.Connection) -> None:
    cols=table_columns(conn,"injury_news_signals")
    if "source_event_id" not in cols: conn.execute("ALTER TABLE injury_news_signals ADD COLUMN source_event_id TEXT")
    if "signal_key" not in cols: conn.execute("ALTER TABLE injury_news_signals ADD COLUMN signal_key TEXT")
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS ux_injury_news_signals_signal_key ON injury_news_signals(signal_key) WHERE signal_key IS NOT NULL")


def backfill_legacy(conn: sqlite3.Connection, matched: pd.DataFrame, season: int, week: int) -> tuple[int,int]:
    rows=conn.execute("SELECT signal_id,gsis_id,source_timestamp,headline FROM injury_news_signals WHERE source=? AND season=? AND week=? AND signal_key IS NULL",(SOURCE,season,week)).fetchall()
    updated=0; unresolved=0
    for signal_id,gsis,ts,headline in rows:
        c=matched[(matched["gsis_id"].eq(clean_text(gsis))) & (matched["source_timestamp"].eq(clean_text(ts))) & (matched["title"].eq(clean_text(headline)))]
        if len(c)==1:
            r=c.iloc[0]
            conn.execute("UPDATE injury_news_signals SET source_event_id=?,signal_key=? WHERE signal_id=? AND signal_key IS NULL",(r["source_event_id"],r["signal_key"],signal_id)); updated+=1
        else: unresolved+=1
    return updated,unresolved


def write_signals(conn: sqlite3.Connection, matched: pd.DataFrame, season: int, week: int) -> tuple[int,int]:
    inserted=existing=0; now=utc_now()
    for row in matched.itertuples(index=False):
        before=conn.total_changes
        conn.execute("""INSERT OR IGNORE INTO injury_news_signals
        (season,week,gsis_id,player_name,team,source,source_timestamp,headline,raw_text,practice_signal,availability_signal,signal_strength,ai_interpretation,ai_confidence,created_at,source_event_id,signal_key)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (season,week,clean_text(row.gsis_id),clean_text(row.player_name),normalize_team(row.team),SOURCE,clean_text(row.source_timestamp),clean_text(row.title),clean_text(row.description),clean_text(row.practice_signal),clean_text(row.availability_signal),clean_text(row.signal_strength),"",None,now,clean_text(row.source_event_id),clean_text(row.signal_key)))
        if conn.total_changes>before: inserted+=1
        else: existing+=1
    return inserted,existing


def main() -> None:
    print("="*72); print("WFS FANDUEL INJURY INGEST V2"); print("="*72)
    if not DB_PATH.exists(): raise RuntimeError(f"Missing database: {DB_PATH}")
    CSV_DIR.mkdir(parents=True,exist_ok=True)
    entities,page_audit=fetch_all_entities()
    print("PAGES =",page_audit["pages"]); print("RAW_ROWS =",page_audit["raw_rows"]); print("BOUNDARY_DUPLICATES =",page_audit["boundary_duplicates"]); print("UNIQUE_SOURCE_EVENTS =",page_audit["unique_events"]); print("TERMINAL_HAS_NEXT_PAGE = False")
    records=extract_records(entities)
    if records.empty or records["source_event_id"].eq("").any() or records["signal_key"].duplicated().any(): raise RuntimeError("Source event identity invariant failed")
    with sqlite3.connect(DB_PATH) as conn:
        season,week=latest_season_week(conn); print(f"WFS target: {season} Week {week}")
        start_utc,end_utc,teams=schedule_window(conn,season,week); print("WINDOW_START_UTC =",start_utc.isoformat()); print("WINDOW_END_UTC =",end_utc.isoformat())
        current,rejected=filter_current_week(records,week,start_utc,end_utc,teams)
        print("CURRENT_WEEK_SOURCE_EVENTS =",len(current)); print("REJECTED_SOURCE_EVENTS =",len(rejected))
        if current.empty: raise RuntimeError("Current-week source filter returned zero rows")
        identity=build_identity_index(conn,season,week); print("Identity rows:",len(identity))
        resolved=attach_identity(current,identity); matched=resolved[resolved["resolution_status"].eq("MATCHED")].copy(); quarantine=resolved[~resolved["resolution_status"].eq("MATCHED")].copy()
        print("MATCHED =",len(matched)); print("QUARANTINED =",len(quarantine))
        if matched.empty or matched["gsis_id"].map(clean_text).eq("").any(): raise RuntimeError("Matched identity invariant failed")
        migrate_schema(conn)
        backfilled,legacy_unresolved=backfill_legacy(conn,matched,season,week); print("LEGACY_BACKFILLED =",backfilled); print("LEGACY_UNRESOLVED =",legacy_unresolved)
        inserted,existing=write_signals(conn,matched,season,week)
        dup=conn.execute("SELECT COUNT(*) FROM (SELECT signal_key FROM injury_news_signals WHERE signal_key IS NOT NULL GROUP BY signal_key HAVING COUNT(*)>1)").fetchone()[0]
        blank_new=conn.execute("SELECT COUNT(*) FROM injury_news_signals WHERE source=? AND season=? AND week=? AND signal_key IS NOT NULL AND (source_event_id IS NULL OR trim(source_event_id)='')",(SOURCE,season,week)).fetchone()[0]
        if dup or blank_new: raise RuntimeError(f"DB event-key invariant failed duplicates={dup} blank_event_ids={blank_new}")
        conn.commit()
    resolved.to_csv(INGEST_CSV,index=False); quarantine.to_csv(QUARANTINE_CSV,index=False)
    print("INSERTED =",inserted); print("ALREADY_PRESENT =",existing); print("SIGNAL_KEY_DUPLICATES = 0"); print("NEW_KEY_ROWS_WITH_BLANK_EVENT_ID = 0")
    if not quarantine.empty: print(quarantine[["player_name","team","position","title","resolution_detail"]].to_string(index=False))
    print("INGEST_CSV =",INGEST_CSV); print("QUARANTINE_CSV =",QUARANTINE_CSV); print("FANDUEL_INJURY_INGEST_V2_STATUS=PASS")

if __name__ == "__main__":
    main()
