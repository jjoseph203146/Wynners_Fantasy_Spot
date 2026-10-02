"""Offline behavior, authority, provenance, and publication adversarial tests."""
import copy
import hashlib
import importlib
from pathlib import Path
import socket
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from shadow_player_analysis_v1.core import SAFETY, consensus
from shadow_player_analysis_v1.fixtures import sample as foundation_sample
from shadow_player_analysis_v1.core import build as foundation_build
from shadow_player_analysis_v1.test_espn_nfl_rss_v1 import feed, item
from shadow_player_analysis_v1.espn_nfl_rss_v1 import ingest
from . import publication
from .adapters import espn_records
from .classification import TAXONOMY, classify
from .core import build
from .fixtures import STORIES, event, sample
from .identity import Authority


class IntelligenceTests(unittest.TestCase):
    def setUp(self):
        self.inputs = sample()
        self.net = patch.object(socket.socket, 'connect', side_effect=AssertionError('NETWORK_FORBIDDEN'))
        self.db = patch.object(sqlite3, 'connect', side_effect=AssertionError('DATABASE_FORBIDDEN'))
        self.net.start()
        self.db.start()
        self.addCleanup(self.net.stop)
        self.addCleanup(self.db.stop)

    def result(self):
        return build(**self.inputs)

    def row(self):
        return self.result()['classified_evidence'][0]

    def players(self):
        return [r for r in self.result()['entity_links'] if r['entity_type'] == 'PLAYER']

    def authority(self):
        return Authority(self.inputs['identity'], self.inputs['schedule'], self.inputs['cutoff'])

    def ambiguous_time(self):
        self.inputs['records'][0].update(published_at_utc=None,
            raw_publication_timestamp='Sun, 27 Sep 2026 15:03:12 EST',
            temporal_status='AMBIGUOUS_SOURCE_TIMESTAMP', temporal_confidence='UNRESOLVED')

    def test_exact_gsis(self):
        r = self.authority().player({'gsis_id':'SYNTHETIC_GSIS_000'})
        self.assertEqual((r['identity_status'], r['gsis_id']), ('RESOLVED','SYNTHETIC_GSIS_000'))

    def test_unknown_id_not_created(self):
        before = copy.deepcopy(self.inputs)
        self.assertIsNone(self.authority().player({'gsis_id':'made-up'})['gsis_id'])
        self.assertEqual(before, self.inputs)

    def test_unresolved(self):
        self.inputs['records'] = [event('Unknown Player will play')]
        self.assertIn('UNRESOLVED_PLAYER_IDENTITY', self.row()['reason_codes'])
        self.assertIsNone(self.players()[0]['gsis_id'])

    def test_ambiguous(self):
        self.inputs['records'] = [event('Alex Common will play')]
        self.assertIn('AMBIGUOUS_PLAYER_IDENTITY', self.row()['reason_codes'])
        self.assertEqual(len(self.players()[0]['candidate_gsis_ids']), 2)

    def test_no_approximate_name(self):
        self.assertIsNone(self.authority().player({'name':'Zay Flowars'})['gsis_id'])

    def test_no_surname_guess(self):
        self.assertIsNone(self.authority().player({'name':'Flowers'})['gsis_id'])

    def test_validated_normalization(self):
        self.assertEqual(self.authority().player({'name':'DE’VON ACHANE'})['identity_status'], 'RESOLVED')

    def test_team_disambiguation_existing_contract(self):
        self.assertEqual(self.authority().player({'name':'Alex Common','team':'CHI'})['gsis_id'], 'SYNTHETIC_GSIS_009')

    def test_context_mismatch(self):
        self.assertIsNone(self.authority().player({'name':'Zay Flowers','team':'CHI'})['gsis_id'])

    def test_gsis_name_mismatch(self):
        self.assertIsNone(self.authority().player({'name':'Case Keenum','gsis_id':'SYNTHETIC_GSIS_000'})['gsis_id'])

    def test_multi_player_independent(self):
        self.inputs['records'] = [event(STORIES[1])]
        links = {r['mention']['name']:r for r in self.players()}
        self.assertEqual(len(links), 2)
        self.assertEqual(links['Case Keenum']['evidence_types'], ['STARTER_CHANGE'])
        self.assertEqual(links['Tyson Bagent']['evidence_types'], ['AVAILABILITY'])
        self.assertEqual(len(self.result()['source_events']), 1)

    def test_team(self):
        self.assertEqual(self.authority().team('Chicago Bears')['team_id'],'CHI')
        self.assertEqual(self.authority().team('LAR')['team_id'],'LA')

    def test_unknown_team(self):
        self.inputs['records'][0]['entities'] = [{'type':'TEAM','name':'Imaginary Team'}]
        self.assertIn('UNRESOLVED_TEAM_IDENTITY',self.row()['reason_codes'])

    def test_game(self):
        self.assertEqual(self.authority().game('SYNTHETIC_GAME_001')['game_id'],'SYNTHETIC_GAME_001')

    def test_unknown_game(self):
        self.inputs['records'][0]['game_id']='invented'
        self.assertIn('UNRESOLVED_GAME_IDENTITY',self.row()['reason_codes'])
        self.assertFalse(self.row()['pregame_eligible'])
        games=[r for r in self.result()['entity_links'] if r['entity_type']=='GAME']
        self.assertIsNone(games[0]['game_id'])

    def test_duplicate_game_fails_closed(self):
        self.inputs['schedule']['games'] *= 2
        self.assertFalse(self.row()['pregame_eligible'])

    def test_no_game_inference(self):
        self.inputs['records'][0].pop('game_id')
        self.assertFalse(self.row()['pregame_eligible'])
        self.assertFalse(any(r['entity_type']=='GAME' for r in self.result()['entity_links']))

    def test_flowers(self):
        self.assertEqual((self.row()['relevance'],self.row()['evidence_types']),('HIGH',['AVAILABILITY']))

    def test_starter(self):
        for i in (1,3):
            with self.subTest(i=i):
                self.assertIn('STARTER_CHANGE',classify(STORIES[i])['evidence_types'])
                self.assertEqual(classify(STORIES[i])['relevance'],'HIGH')

    def test_inactive(self):
        self.assertEqual(classify(STORIES[2])['evidence_types'],['AVAILABILITY'])

    def test_workload_increase(self):
        self.assertIn('ROLE_INCREASE',classify(STORIES[12])['evidence_types'])

    def test_workload_decrease(self):
        self.assertIn('ROLE_DECREASE',classify(STORIES[13])['evidence_types'])

    def test_coach(self):
        self.assertIn('COACH_INTENT',classify(STORIES[14])['evidence_types'])

    def test_defender_environment(self):
        for i in (5,6):
            with self.subTest(i=i):
                self.inputs['records']=[event(STORIES[i])]
                self.assertIn('TEAM_ENVIRONMENT',self.row()['evidence_types'])
                self.assertIn('INJURY_CONTEXT',self.players()[0]['evidence_types'])
                self.assertEqual(self.players()[0]['identity_status'],'RESOLVED')

    def test_uniforms(self):
        self.assertEqual((classify(STORIES[7])['relevance'],classify(STORIES[7])['evidence_types']),('REJECT',['NON_FOOTBALL']))

    def test_fashion(self):
        self.assertEqual(classify(STORIES[8])['evidence_types'],['NON_FOOTBALL'])

    def test_college(self):
        self.assertEqual(classify(STORIES[9])['evidence_types'],['NON_CURRENT_NFL'])

    def test_market_not_fact(self):
        self.assertEqual(classify(STORIES[10])['evidence_kind'],'MARKET_INFORMATION')
        self.assertEqual(classify(STORIES[10])['evidence_types'],['MARKET_CONTEXT'])

    def test_recap(self):
        self.assertEqual(classify(STORIES[11])['evidence_types'],['POST_GAME_RECAP'])

    def test_recap_cannot_create_forward_role(self):
        self.assertEqual(classify(STORIES[11]+'; expanded role')['evidence_types'],['POST_GAME_RECAP'])

    def test_generic_box_words(self):
        for word in ['carries','targets','catches','yards','snaps']:
            with self.subTest(word=word):
                self.assertEqual(classify('Zay Flowers '+word)['evidence_types'],['INSUFFICIENT'])

    def test_factual(self):
        self.assertEqual(classify(STORIES[2])['evidence_kind'],'FACTUAL_OBSERVATION')

    def test_expectation(self):
        self.assertEqual(classify(STORIES[3])['evidence_kind'],'REPORTED_EXPECTATION')

    def test_opinion(self):
        self.assertEqual(classify('Analyst likes Zay Flowers')['evidence_kind'],'ANALYST_OPINION')

    def test_opinion_no_role(self):
        self.assertEqual(classify('Analyst predicts Zay Flowers expanded role')['evidence_types'],['DFS_ANALYSIS'])

    def test_negation(self):
        self.assertEqual(classify('Zay Flowers will not have an expanded role')['evidence_types'],['INSUFFICIENT'])

    def test_question(self):
        self.assertEqual(classify('Will Zay Flowers have an expanded role?')['evidence_types'],['INSUFFICIENT'])

    def test_temporal_ambiguous_classifies(self):
        self.ambiguous_time()
        self.assertEqual(self.row()['evidence_types'],['AVAILABILITY'])

    def test_temporal_ambiguous_resolves(self):
        self.ambiguous_time()
        self.assertEqual(self.players()[0]['identity_status'],'RESOLVED')

    def test_temporal_ambiguous_no_pregame(self):
        self.ambiguous_time()
        self.assertFalse(self.row()['pregame_eligible'])
        self.assertIn('TEMPORAL_UNRESOLVED',self.row()['reason_codes'])

    def test_temporal_ambiguous_no_claim(self):
        self.ambiguous_time()
        self.assertFalse(self.row()['claim_eligible'])
        self.assertTrue(all(not r['claim_eligible'] for r in self.result()['entity_links']))

    def test_temporal_ambiguous_no_vote(self):
        self.ambiguous_time()
        self.assertFalse(self.row()['consensus_eligible'])
        self.assertEqual(self.result()['source_quality'][0]['consensus_votes'],0)

    def test_post_kickoff(self):
        r=self.inputs['records'][0]
        r.update(published_at_utc='2025-09-07T18:00:00Z',first_seen_at_utc='2025-09-07T18:01:00Z',retrieved_at_utc='2025-09-07T18:02:00Z')
        self.assertIn('POST_KICKOFF_FOR_PREGAME',self.row()['reason_codes'])
        self.assertFalse(self.row()['pregame_eligible'])

    def test_retrieval_not_publication(self):
        self.ambiguous_time()
        self.result()
        self.assertIsNone(self.result()['source_events'][0]['published_at_utc'])

    def test_first_seen_not_publication(self):
        self.inputs['records'][0]['published_at_utc']=None
        self.assertFalse(self.row()['pregame_eligible'])
        self.assertIsNone(self.result()['source_events'][0]['published_at_utc'])

    def test_valid_temporal(self):
        self.assertTrue(self.row()['pregame_eligible'])
        self.assertFalse(self.row()['claim_eligible'])

    def test_late_knowledge(self):
        self.inputs['records'][0]['retrieved_at_utc']='2025-09-07T16:30:00Z'
        self.assertIn('EVIDENCE_AFTER_CUTOFF',self.row()['reason_codes'])

    def test_invalid_time_order(self):
        self.inputs['records'][0]['first_seen_at_utc']='2025-09-07T09:00:00Z'
        self.assertIn('INVALID_SOURCE_TIME_ORDER',self.row()['reason_codes'])

    def test_timezone_required(self):
        self.inputs['records'][0]['published_at_utc']='2025-09-07T10:00:00'
        self.assertIn('TEMPORAL_UNRESOLVED',self.row()['reason_codes'])

    def test_provenance_all_fields(self):
        original=copy.deepcopy(self.inputs['records'][0])
        row=self.result()['source_events'][0]
        for k,v in original.items():
            with self.subTest(field=k):
                self.assertEqual(row[k],v)

    def test_guid_and_raw_feed_preserved(self):
        capture=ingest(feed(item(title=STORIES[0],pub='Sun, 07 Sep 2025 10:00:00 EST')),'2025-09-07T16:00:00Z')
        records=espn_records(capture)
        self.inputs['records']=records
        self.inputs['approvals']={'ESPN_NFL_RSS_V1':['ESPN_RSS_SHADOW_CAPTURE_ONLY']}
        row=self.result()['source_events'][0]
        self.assertEqual(row['source_event_id'],'guid:stable-1')
        self.assertEqual(row['raw_feed_fields']['guid'],'stable-1')
        self.assertEqual(row['raw_publication_timestamp'],'Sun, 07 Sep 2025 10:00:00 EST')
        self.assertEqual(row['article_url'],capture['quarantine'][0]['article_url'])
        self.assertEqual(row['content_hash'],capture['quarantine'][0]['content_hash'])
        self.assertEqual(row['origin_id'],'ESPN')
        self.assertFalse(self.row()['pregame_eligible'])
        self.assertEqual(self.players()[0]['identity_status'],'RESOLVED')

    def test_espn_events_quarantine_not_duplicated(self):
        capture=ingest(feed(item(title=STORIES[0])),'2025-09-07T16:00:00Z')
        self.assertEqual(len(espn_records(capture)),1)

    def test_adapter_failures_preserved(self):
        self.inputs['records'][0]['adapter_reason_codes']=['DUPLICATE_ITEM_FIELD']
        self.assertIn('DUPLICATE_ITEM_FIELD',self.row()['reason_codes'])

    def test_duplicate_event(self):
        original=self.result()
        self.inputs['records']*=2
        self.assertEqual(self.result(),original)

    def test_order_independent_build(self):
        self.inputs=sample(True)
        original=self.result()
        self.inputs['records'].reverse()
        self.assertEqual(original,self.result())

    def test_revision_conflict(self):
        r=event('Zay Flowers inactive')
        self.inputs['records'].append(r)
        self.assertTrue(all('SOURCE_EVENT_REVISION_CONFLICT' in r['reason_codes'] for r in self.result()['classified_evidence']))

    def test_same_origin_no_votes(self):
        self.inputs['records'].append(dict(event(),source_event_id='other'))
        self.assertEqual(self.result()['source_quality'][0]['consensus_votes'],0)
        self.assertEqual(len(self.result()['source_quality'][0]['origins']),1)

    def test_syndication_no_votes(self):
        self.inputs['records'].append(dict(event(),source='SYNTHETIC_B'))
        self.assertEqual(sum(r['consensus_votes'] for r in self.result()['source_quality']),0)

    def test_foundation_conflict_preserved(self):
        row=foundation_build(foundation_sample())['claims'][0]
        contrary=dict(row,claim_id='contrary',origin_id='independent',direction='DECREASE',signal_type='ROLE_DECREASE')
        self.assertEqual(consensus([row,contrary])[0]['status'],'CONFLICT')

    def test_conflicting_evidence_retained(self):
        self.inputs['records']=[event('Zay Flowers workload will increase next game'),
            dict(event('Zay Flowers workload will decrease next game',1),origin_id='independent')]
        self.assertEqual({tuple(r['evidence_types']) for r in self.result()['classified_evidence']},{('ROLE_INCREASE',),('ROLE_DECREASE',)})
        self.assertTrue(all(not r['consensus_eligible'] for r in self.result()['classified_evidence']))

    def test_unresolved_no_vote(self):
        self.inputs['records']=[event('Unknown Player will play')]
        self.assertFalse(self.players()[0]['consensus_eligible'])

    def test_rejected_no_vote(self):
        self.inputs['records']=[event(STORIES[8])]
        self.assertFalse(self.row()['consensus_eligible'])
        self.assertIn('REJECTED_CONTENT',self.row()['reason_codes'])

    def test_unapproved_independent_gate(self):
        self.inputs['approvals']={}
        self.assertIn('SOURCE_NOT_APPROVED',self.row()['reason_codes'])
        self.assertEqual(self.players()[0]['identity_status'],'RESOLVED')
        self.assertTrue(self.row()['pregame_eligible'])

    def test_multiple_reasons(self):
        self.ambiguous_time()
        self.inputs['approvals']={}
        self.inputs['records'][0]['entities']=[{'type':'PLAYER','name':'Unknown Player'}]
        self.assertTrue({'SOURCE_NOT_APPROVED','TEMPORAL_UNRESOLVED','UNRESOLVED_PLAYER_IDENTITY'} <= set(self.row()['reason_codes']))

    def test_snapshot_not_mutated(self):
        before=copy.deepcopy(self.inputs)
        self.result()
        self.assertEqual(self.inputs,before)

    def test_future_authority_rejected(self):
        self.inputs['identity']['available_at_utc']='2025-09-08T00:00:00Z'
        with self.assertRaisesRegex(ValueError,'AUTHORITY_FROM_FUTURE'):
            self.result()

    def test_wrong_authority_rejected(self):
        self.inputs['identity']['authority']='ESPN_NEW_IDENTITIES'
        with self.assertRaisesRegex(ValueError,'IDENTITY_AUTHORITY'):
            self.result()

    def test_conflicting_snapshot_rejected(self):
        self.inputs['identity']['players'].append(dict(self.inputs['identity']['players'][0],team='CHI'))
        with self.assertRaisesRegex(ValueError,'CONFLICTING_IDENTITY_SNAPSHOT'):
            self.result()

    def test_safety(self):
        self.inputs=sample(True)
        for name in ('classified_evidence','entity_links','quarantine'):
            for row in self.result()[name]:
                self.assertEqual(row['safety'],SAFETY)

    def test_no_network_or_db(self):
        self.inputs=sample(True)
        self.result()
        self.assertTrue(True)  # setUp blocks real socket connections and all SQLite connections.

    def test_import_no_io(self):
        from . import identity,core,adapters,classification
        for mod in (identity,core,adapters,classification,publication):
            importlib.reload(mod)

    def test_all_fixtures_and_taxonomy(self):
        self.inputs=sample(True)
        result=self.result()
        self.assertEqual(len(STORIES),20)
        self.assertEqual(len(result['source_events']),19)
        for r in result['classified_evidence']:
            self.assertTrue(set(r['evidence_types']) <= TAXONOMY)

    def test_taxonomy_routes(self):
        for token in ('target share','route participation','backfield share','red zone role','deep target role',
                      'usage analysis','favorable matchup','unfavorable matchup','coverage matchup',
                      'pass rush','game script','team environment','signed','DFS analysis'):
            with self.subTest(token=token):
                self.assertNotEqual(classify('Zay Flowers '+token)['evidence_types'],['INSUFFICIENT'])

    def test_unanchored_quote_no_entity_role(self):
        self.inputs['records'][0]['entities']=[{'type':'PLAYER','name':'Case Keenum','evidence_quote':STORIES[0]}]
        link=next(r for r in self.players() if r['mention']['name']=='Case Keenum')
        self.assertEqual(link['evidence_types'],['INSUFFICIENT'])

    def test_coordinated_players_no_role_assignment(self):
        self.inputs['records']=[event('Case Keenum replaces Tyson Bagent as starting quarterback')]
        self.assertTrue(all(r['evidence_types']==['INSUFFICIENT'] for r in self.players()))

    def test_supplied_opinion_cannot_become_role(self):
        self.inputs['records']=[dict(event(STORIES[12]), evidence_kind='ANALYST_OPINION')]
        self.assertEqual(self.row()['evidence_types'],['DFS_ANALYSIS'])
        self.assertTrue(all(r['evidence_types']==['DFS_ANALYSIS'] for r in self.players()))

    def test_supplied_market_preserved(self):
        self.inputs['records'][0]['evidence_kind']='MARKET_INFORMATION'
        self.assertEqual(self.row()['evidence_kind'],'MARKET_INFORMATION')
        self.assertEqual(self.row()['evidence_types'],['MARKET_CONTEXT'])

    def test_supplied_expectation_preserved(self):
        self.inputs['records'][0]['evidence_kind']='REPORTED_EXPECTATION'
        self.assertEqual(self.row()['evidence_kind'],'REPORTED_EXPECTATION')

    def test_wfs_corroboration_is_not_claim_authority(self):
        self.inputs['records'][0]['evidence_kind']='WFS_CORROBORATION'
        self.assertEqual(self.row()['evidence_kind'],'WFS_CORROBORATION')
        self.assertFalse(self.row()['claim_eligible'])

    def test_factual_label_cannot_override_expectation(self):
        self.inputs['records']=[dict(event(STORIES[3]), evidence_kind='FACTUAL_OBSERVATION')]
        self.assertEqual(self.row()['evidence_kind'],'REPORTED_EXPECTATION')

    def test_recap_never_pregame(self):
        self.inputs['records']=[event(STORIES[11])]
        self.assertFalse(self.row()['pregame_eligible'])

    def test_later_player_relationship_retained(self):
        self.inputs['records']=[event('Zay Flowers will play; Zay Flowers workload will increase next game')]
        self.assertEqual({tuple(r['evidence_types']) for r in self.players()}, {('AVAILABILITY',),('ROLE_INCREASE',)})

    def test_team_not_unknown_player(self):
        self.inputs['records']=[event('Chicago Bears will play')]
        self.assertEqual(self.players(),[])

    def test_malformed_provenance_fails_before_publication(self):
        self.inputs['records'][0].pop('raw_publication_timestamp')
        with self.assertRaisesRegex(ValueError,'MISSING_PUBLICATION_PROVENANCE'):
            self.result()

    def test_adapter_cannot_assign_whole_multi_player_quote(self):
        text='Case Keenum replaces Tyson Bagent as starting quarterback'
        self.inputs['records']=[dict(event(text), entities=[dict(type='PLAYER',name='Tyson Bagent',evidence_quote=text)])]
        self.assertTrue(all(r['evidence_types']==['INSUFFICIENT'] for r in self.players()))


    def test_same_event_unique_surname_starter_link(self):
        text = (
            "With Winston set to start, Giants mull options for additions at QB\n"
            "Jameis Winston performs against the Titans in Week 3."
        )
        self.inputs['records'] = [event(text)]
        links = [
            r for r in self.players()
            if r['gsis_id'] == 'SYNTHETIC_GSIS_004'
        ]
        self.assertTrue(links)
        self.assertTrue(
            any(r['evidence_types'] == ['STARTER_CHANGE'] for r in links)
        )
        self.assertNotIn(
            'UNRESOLVED_PLAYER_IDENTITY',
            self.row()['reason_codes']
        )

    def test_same_event_surname_without_full_name_fails_closed(self):
        self.inputs['records'] = [event('Winston set to start Sunday')]
        self.assertIn(
            'NO_SUPPORTED_ENTITY',
            self.row()['reason_codes']
        )
        self.assertFalse(
            any(r['gsis_id'] == 'SYNTHETIC_GSIS_004'
                for r in self.players())
        )

    def test_same_event_ambiguous_surname_fails_closed(self):
        self.inputs['identity']['players'].append({
            'gsis_id': 'SYNTHETIC_GSIS_MARCUS_WINSTON',
            'player_name': 'Marcus Winston',
            'team': 'TEN',
            'position': 'QB',
        })
        text = (
            "Winston set to start Sunday\n"
            "Jameis Winston and Marcus Winston were discussed."
        )
        self.inputs['records'] = [event(text)]
        self.assertTrue(
            all(r['evidence_types'] == ['INSUFFICIENT']
                for r in self.players()
                if r['gsis_id'] in {
                    'SYNTHETIC_GSIS_004',
                    'SYNTHETIC_GSIS_MARCUS_WINSTON'
                })
        )
        winston_links = [
            r for r in self.players()
            if r['gsis_id'] in {
                'SYNTHETIC_GSIS_004',
                'SYNTHETIC_GSIS_MARCUS_WINSTON'
            }
        ]
        self.assertTrue(winston_links)
        self.assertTrue(
            all('STARTER_CHANGE' not in r['evidence_types']
                for r in winston_links)
        )

    def test_same_event_negated_start_not_promoted(self):
        text = (
            "Winston not expected to start Sunday\n"
            "Jameis Winston remains with the Giants."
        )
        self.inputs['records'] = [event(text)]
        links = [
            r for r in self.players()
            if r['gsis_id'] == 'SYNTHETIC_GSIS_004'
        ]
        self.assertTrue(
            all('STARTER_CHANGE' not in r['evidence_types']
                for r in links)
        )



class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.inputs=sample(True)
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)/'shadow_player_analysis_v1_2'
        self.root.mkdir()
        self.root_patch=patch.object(publication,'ROOT',self.root)
        self.out_patch=patch.object(publication,'OUTPUT',self.root/'artifacts')
        self.root_patch.start(); self.out_patch.start()
        self.addCleanup(self.root_patch.stop); self.addCleanup(self.out_patch.stop)
        # bundle dependency hashing uses the actual code root, while publication
        # path tests are confined to an isolated temporary directory.
        original_bundle=publication.bundle
        actual_root=Path(__file__).resolve().parent
        def bundle(inputs):
            with patch.object(publication,'ROOT',actual_root):
                return original_bundle(inputs)
        self.bundle_patch=patch.object(publication,'bundle',side_effect=bundle)
        self.bundle_patch.start(); self.addCleanup(self.bundle_patch.stop)

    def test_deterministic(self):
        self.assertEqual(publication.bundle(self.inputs),publication.bundle(copy.deepcopy(self.inputs)))

    def test_immutable_history(self):
        first=publication.publish(self.inputs)
        before={p.name:p.read_bytes() for p in first.iterdir()}
        self.inputs['records'].append(event('Zay Flowers inactive',100))
        second=publication.publish(self.inputs)
        self.assertNotEqual(first,second)
        self.assertEqual(before,{p.name:p.read_bytes() for p in first.iterdir()})

    def test_idempotent(self):
        first=publication.publish(self.inputs)
        self.assertEqual(first,publication.publish(self.inputs))

    def test_manifest_hashes(self):
        path=publication.publish(self.inputs)
        manifest=publication.verify(path)
        for name,meta in manifest['files'].items():
            self.assertEqual(meta['sha256'],hashlib.sha256((path/name).read_bytes()).hexdigest())

    def test_failed_publication_no_partial(self):
        with patch.object(publication.os,'rename',side_effect=OSError('simulated')):
            with self.assertRaises(OSError):
                publication.publish(self.inputs)
        self.assertEqual(list(publication.OUTPUT.iterdir()),[])

    def test_failed_write_no_partial(self):
        with patch.object(publication.os,'fsync',side_effect=OSError('simulated write failure')):
            with self.assertRaises(OSError):
                publication.publish(self.inputs)
        self.assertEqual(list(publication.OUTPUT.iterdir()),[])

    def test_tamper_detected(self):
        path=publication.publish(self.inputs)
        (path/'entity_links.json').write_text('tampered')
        with self.assertRaisesRegex(ValueError,'ARTIFACT_HASH'):
            publication.publish(self.inputs)

    def test_symlink_rejected(self):
        target=self.root/'elsewhere'; target.mkdir()
        publication.OUTPUT.symlink_to(target,target_is_directory=True)
        with self.assertRaisesRegex(ValueError,'OUTPUT_PATH'):
            publication.publish(self.inputs)

    def test_no_mutable_pointer(self):
        path=publication.publish(self.inputs)
        self.assertEqual(list(publication.OUTPUT.iterdir()),[path])
        self.assertEqual(set(p.name for p in path.iterdir()),publication.FILES|{'manifest.json'})

    def test_publication_no_db_or_network(self):
        with patch.object(sqlite3,'connect',side_effect=AssertionError('DB')), patch.object(socket.socket,'connect',side_effect=AssertionError('NET')):
            publication.verify(publication.publish(self.inputs))


if __name__ == '__main__':
    unittest.main()
