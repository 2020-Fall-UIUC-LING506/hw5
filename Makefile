all: data/tiny.txt data/1984.txt

data/tiny.txt: data/1984.txt | data
	head -n 1000 $< | tail -n 100 > $@

data/1984.txt: | data 
	curl http://gutenberg.net.au/ebooks01/0100021.txt | perl -ne 's/\r//; print $$_;' | scripts/split-sentences.perl | grep -v '^<' | scripts/tokenizer.perl > $@

data:
	mkdir $@

clean:
	rm data/*.txt
