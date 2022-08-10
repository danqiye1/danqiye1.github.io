all:
	bundle exec jekyll serve
	
build:
	bundle install

clean:
	rm -rf _site