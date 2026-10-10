# AUTO-GENERATED: Do not edit manually.

from ..models.v1.annotation import Annotation
from ..models.v1.confidence.agreement import ConfidenceByInterAnnotatorAgreement
from ..models.v1.confidence.likelihood import LikelihoodConfidence
from ..models.v1.confidence.variance import VarianceConfidence
from ..models.v1.extent.midi_interval import MidiInterval
from ..models.v1.extent.midi_ticks import MidiTicks
from ..models.v1.extent.pixel_box import PixelBoxExtent
from ..models.v1.extent.quarters_interval_float import QuartersIntervalFloat
from ..models.v1.extent.quarters_interval_fraction import QuartersIntervalFraction
from ..models.v1.extent.quarters_time_float import QuartersTimeFloat
from ..models.v1.extent.quarters_time_fraction import QuartersTimeFraction
from ..models.v1.extent.time_frequency_box import TimeFrequencyBoxExtent
from ..models.v1.extent.time_interval import TimeIntervalExtent
from ..models.v1.extent.times import Times
from ..models.v1.metadata.algorithm import AlgorithmAnnotationMetadata
from ..models.v1.metadata.crowd import CrowdSourcedAnnotationMetadata
from ..models.v1.metadata.derived import DerivedAnnotationMetadata
from ..models.v1.metadata.human import HumanAnnotationMetadata
from ..models.v1.metadata.other import AnnotationMetadataOther
from ..models.v1.metadata.sensor import SensorAnnotationMetadata
from ..models.v1.payload.beat import BeatPositionPayload
from ..models.v1.payload.chord import ChordPayload
from ..models.v1.payload.ext import ExtensionPayload
from ..models.v1.payload.key_mode import KeyModePayload
from ..models.v1.payload.lyrics import LyricsPayload
from ..models.v1.payload.mood_thayer import MoodThayerPayload
from ..models.v1.payload.note_hz import NoteHzPayload
from ..models.v1.payload.note_midi import NoteMidiPayload
from ..models.v1.payload.object import ObjectPayload
from ..models.v1.payload.onset import OnsetPayload
from ..models.v1.payload.pitch_contour_hz import PitchContourPayload
from ..models.v1.payload.relation import RelationPayload
from ..models.v1.payload.score_control_event.chord import ScoreChordPayload
from ..models.v1.payload.score_control_event.dynamic import ScoreDynamicPayload
from ..models.v1.payload.score_control_event.figured_bass import ScoreFiguredBassPayload
from ..models.v1.payload.score_control_event.spanner import ScoreSpannerPayload
from ..models.v1.payload.score_control_event.staff_text import ScoreStaffTextPayload
from ..models.v1.payload.score_control_event.system_text import ScoreSystemTextPayload
from ..models.v1.payload.score_control_event.tempo import ScoreTempoPayload
from ..models.v1.payload.score_note import ScoreNotePayload
from ..models.v1.payload.segment_multi import MultiSegmentPayload
from ..models.v1.payload.segment_open import SegmentOpenPayload
from ..models.v1.payload.tag_open import TagOpenPayload
from ..models.v1.payload.tempo import TempoPayload

ANNOTATION_CLASS = Annotation

CONFIDENCE_TYPE_REGISTRY = {
    'agreement': ConfidenceByInterAnnotatorAgreement,
    'likelihood': LikelihoodConfidence,
    'variance': VarianceConfidence,
}

EXTENT_TYPE_REGISTRY = {
    'midi_interval': MidiInterval,
    'midi_ticks': MidiTicks,
    'pixel_box': PixelBoxExtent,
    'quarters_interval.float': QuartersIntervalFloat,
    'quarters_interval.fraction': QuartersIntervalFraction,
    'quarters_time.float': QuartersTimeFloat,
    'quarters_time.fraction': QuartersTimeFraction,
    'time': Times,
    'time_frequency_box': TimeFrequencyBoxExtent,
    'time_interval': TimeIntervalExtent,
}

METADATA_TYPE_REGISTRY = {
    'algorithm': AlgorithmAnnotationMetadata,
    'crowd': CrowdSourcedAnnotationMetadata,
    'derived': DerivedAnnotationMetadata,
    'human': HumanAnnotationMetadata,
    'other': AnnotationMetadataOther,
    'sensor': SensorAnnotationMetadata,
}

PAYLOAD_TYPE_REGISTRY = {
    'beat': BeatPositionPayload,
    'chord': ChordPayload,
    'ext': ExtensionPayload,
    'key_mode': KeyModePayload,
    'lyrics': LyricsPayload,
    'mood_thayer': MoodThayerPayload,
    'multi_segment': MultiSegmentPayload,
    'note_hz': NoteHzPayload,
    'note_midi': NoteMidiPayload,
    'object': ObjectPayload,
    'onset': OnsetPayload,
    'pitch_contour': PitchContourPayload,
    'relation': RelationPayload,
    'score_control_event.chord': ScoreChordPayload,
    'score_control_event.dynamic': ScoreDynamicPayload,
    'score_control_event.figured_bass': ScoreFiguredBassPayload,
    'score_control_event.spanner': ScoreSpannerPayload,
    'score_control_event.staff_text': ScoreStaffTextPayload,
    'score_control_event.system_text': ScoreSystemTextPayload,
    'score_control_event.tempo': ScoreTempoPayload,
    'score_note': ScoreNotePayload,
    'segment_open': SegmentOpenPayload,
    'tag_open': TagOpenPayload,
    'tempo': TempoPayload,
}

COMPLEX_FIELDS_REGISTRY: dict[str, dict[str, list[str]]] = {
    'confidence_type': {
        'agreement': ['n_annotators'],
    },
    'extent_type': {
        'quarters_interval.fraction': ['quarter', 'duration'],
        'quarters_time.fraction': ['quarter'],
    },
    'metadata_type': {
        'algorithm': ['parameters'],
        'derived': ['parameters'],
        'sensor': ['settings'],
    },
    'payload_type': {
        'ext': ['value'],
        'object': ['value'],
        'onset': ['value'],
        'pitch_contour': ['value'],
        'relation': ['relation'],
        'score_control_event.chord': ['volta', 'slur', 'crescendo_hairpin', 'decrescendo_hairpin', 'crescendo_line', 'diminuendo_line', 'pedal', 'nominal_duration', 'scalar', 'gracenote', 'articulation', 'tremolo', 'lyrics_1', 'lyrics_2', 'lyrics_3'],
        'score_control_event.dynamic': ['volta', 'slur', 'crescendo_hairpin', 'decrescendo_hairpin', 'crescendo_line', 'diminuendo_line', 'pedal'],
        'score_control_event.figured_bass': ['thoroughbass_duration', 'volta', 'slur', 'crescendo_hairpin', 'decrescendo_hairpin', 'crescendo_line', 'diminuendo_line', 'pedal', 'thoroughbass_level_1', 'thoroughbass_level_2', 'thoroughbass_level_3', 'thoroughbass_level_4'],
        'score_control_event.spanner': ['volta', 'slur', 'crescendo_hairpin', 'decrescendo_hairpin', 'crescendo_line', 'diminuendo_line', 'pedal'],
        'score_control_event.staff_text': ['volta', 'slur', 'crescendo_hairpin', 'decrescendo_hairpin', 'crescendo_line', 'diminuendo_line', 'pedal', 'staff_text'],
        'score_control_event.system_text': ['volta', 'slur', 'crescendo_hairpin', 'decrescendo_hairpin', 'crescendo_line', 'diminuendo_line', 'pedal'],
        'score_control_event.tempo': ['volta', 'slur', 'crescendo_hairpin', 'decrescendo_hairpin', 'crescendo_line', 'diminuendo_line', 'pedal', 'tempo', 'metronome_base', 'metronome_number'],
        'score_note': ['name', 'octave', 'staff', 'voice', 'mc', 'mn', 'measure', 'tied', 'gracenote', 'nominal_duration', 'scalar', 'chord_id', 'tremolo', 'volta', 'tuning', 'anchor'],
    },
}

__all__ = ['ANNOTATION_CLASS', 'COMPLEX_FIELDS_REGISTRY', 'CONFIDENCE_TYPE_REGISTRY', 'EXTENT_TYPE_REGISTRY', 'METADATA_TYPE_REGISTRY', 'PAYLOAD_TYPE_REGISTRY']
