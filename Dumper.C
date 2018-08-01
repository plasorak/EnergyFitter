



void Dumper() {

  TFile *f = new TFile("AlexOutput.root", "READ");
  int ChanWidth,Type,NHits,Config,Event;
  float SumADC,TimeWidth;
  std::vector<double>* ENu = NULL;

  
  TTree* tree = (TTree*)f->Get("ClusteredWireHit");
  tree->SetBranchAddress("Config", &Config   );
  tree->SetBranchAddress("ChanWidth", &ChanWidth);
  tree->SetBranchAddress("Event", &Event);
  tree->SetBranchAddress("Type", &Type     );
  tree->SetBranchAddress("NHits", &NHits    );
  tree->SetBranchAddress("SumADC", &SumADC   );
  tree->SetBranchAddress("TimeWidth", &TimeWidth);

  TTree* truetree = (TTree*)f->Get("TrueInfo");
  truetree->SetBranchAddress("Event", &Event);
  truetree->SetBranchAddress("ENu",   &ENu  );
  
  std::map<int,double> map_evnumber_enu;
  
  for(int j=0;j<truetree->GetEntries();++j) {
    truetree->GetEntry(j);
    map_evnumber_enu[Event] = ENu->front();
  }

  ofstream myfile;
  for(int i=0; i<6; ++i) {
    std::cout << "Doing config " << i << std::endl;
    int nSign=0;
    int nBack=0;
    myfile.open (Form("Config%i.txt", i));
    myfile << "Event,ChanWidth,Type,NHits,SumADC,TimeWidth,ENu"<<std::endl;
    for(int j=0;j<tree->GetEntries();++j) {
      if(j%100000==0)
        std::cout << "did " << 100.*j/tree->GetEntries() << std::endl;
      tree->GetEntry(j);
      if(Config!=i)
        continue;
      myfile << Event<<","<<ChanWidth<<","<<Type<<","<<NHits<<","<<SumADC<<","<<TimeWidth<<","<<map_evnumber_enu[Event]*1000<<std::endl;
      nSign+=Type;
      nBack++;
    }
    nBack-=nSign;
    myfile.close();
    std::cout << "nSign " << nSign << std::endl;
    std::cout << "nBack " << nBack << std::endl;

  }
}


