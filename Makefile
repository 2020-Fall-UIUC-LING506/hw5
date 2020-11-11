data/1984.txt: 
	mkdir -f data ; curl http://gutenberg.net.au/ebooks01/0100021.txt | perl -ne 's/\r//; print $$_;' | scripts/split-sentences.perl | grep -v '^<' | scripts/tokenizer.perl > $@


clean:
	rm data/*.txt
