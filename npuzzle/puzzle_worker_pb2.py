# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: puzzle_worker.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='puzzle_worker.proto',
  package='helloworld',
  syntax='proto3',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\x13puzzle_worker.proto\x12\nhelloworld\"\x1f\n\rPuzzleRequest\x12\x0e\n\x06puzzle\x18\x01 \x01(\t\"\x1b\n\x0bPuzzleReply\x12\x0c\n\x04path\x18\x01 \x01(\t2M\n\x07Puzzler\x12\x42\n\nFindPuzzle\x12\x19.helloworld.PuzzleRequest\x1a\x17.helloworld.PuzzleReply\"\x00\x62\x06proto3'
)




_PUZZLEREQUEST = _descriptor.Descriptor(
  name='PuzzleRequest',
  full_name='helloworld.PuzzleRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='puzzle', full_name='helloworld.PuzzleRequest.puzzle', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=35,
  serialized_end=66,
)


_PUZZLEREPLY = _descriptor.Descriptor(
  name='PuzzleReply',
  full_name='helloworld.PuzzleReply',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='path', full_name='helloworld.PuzzleReply.path', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=68,
  serialized_end=95,
)

DESCRIPTOR.message_types_by_name['PuzzleRequest'] = _PUZZLEREQUEST
DESCRIPTOR.message_types_by_name['PuzzleReply'] = _PUZZLEREPLY
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

PuzzleRequest = _reflection.GeneratedProtocolMessageType('PuzzleRequest', (_message.Message,), {
  'DESCRIPTOR' : _PUZZLEREQUEST,
  '__module__' : 'puzzle_worker_pb2'
  # @@protoc_insertion_point(class_scope:helloworld.PuzzleRequest)
  })
_sym_db.RegisterMessage(PuzzleRequest)

PuzzleReply = _reflection.GeneratedProtocolMessageType('PuzzleReply', (_message.Message,), {
  'DESCRIPTOR' : _PUZZLEREPLY,
  '__module__' : 'puzzle_worker_pb2'
  # @@protoc_insertion_point(class_scope:helloworld.PuzzleReply)
  })
_sym_db.RegisterMessage(PuzzleReply)



_PUZZLER = _descriptor.ServiceDescriptor(
  name='Puzzler',
  full_name='helloworld.Puzzler',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_start=97,
  serialized_end=174,
  methods=[
  _descriptor.MethodDescriptor(
    name='FindPuzzle',
    full_name='helloworld.Puzzler.FindPuzzle',
    index=0,
    containing_service=None,
    input_type=_PUZZLEREQUEST,
    output_type=_PUZZLEREPLY,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
])
_sym_db.RegisterServiceDescriptor(_PUZZLER)

DESCRIPTOR.services_by_name['Puzzler'] = _PUZZLER

# @@protoc_insertion_point(module_scope)
