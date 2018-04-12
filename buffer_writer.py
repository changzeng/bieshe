# encoding: utf-8
# author: liaochangzeng
# e-mail: 1207338829@qq.com


class BufferWriter(object):
	def __init__(self, file_name, max_buffer_size=30*1024*1024):
		self.file_name = file_name
		self.fd = open(self.file_name, "w+")
		self.records = []
		self.records_size = 0
		self.max_buffer_size = max_buffer_size

	def update(self, record):
		self.records.append(record)
		self.records_size += len(record)

		if self.records_size >= self.max_buffer_size:
			self.write_to_file()
			self.records = []
			self.records_size = 0

	def write_to_file(self):
		print("writting to %s" % self.file_name)
		self.fd.write("\n".join(self.records))
		self.fd.write("\n")

	def close(self):
		if self.records_size > 0:
			self.write_to_file()
		self.fd.close()